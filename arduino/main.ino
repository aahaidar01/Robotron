#include <Arduino.h>
#include <PortentaEthernet.h>
#include <Ethernet.h>
#include "config.h"        // Shared constants + extern Print* logOut
#include "q_table.h"       // Your 2048x3 exported Q-table
#include "lidar_sensor.h"  // LiDAR, Target Math, Reflexes
#include "chassis.h"       // Motors, Odom, Sensor Fusion

// ================== ETHERNET CONFIG ==================
// Static IP for the Portenta. Make sure your PC's Ethernet adapter
// is on the same subnet (e.g. 192.168.1.100, netmask 255.255.255.0).
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
IPAddress ip(192, 168, 1, 177);
const int ETH_PORT = 23;

EthernetServer server(ETH_PORT);
EthernetClient ethClient;

// Global output stream (defined here, declared extern in config.h).
// Starts on USB Serial, switches to Ethernet once a TCP client connects.
Print* logOut = &Serial;

// ================== ROBOT STATE MACHINE ==================
enum RobotState { WAIT_CLIENT, WAIT_TRIGGER, INITIALIZING, RUNNING };
static RobotState robotState = WAIT_CLIENT;


/* ========================================================================
   RL AGENT CONFIGURATION
   ======================================================================== */
const int NUM_ACTIONS = 3;
const char* ACTION_NAMES[] = {"FWD", "LFT", "RGT"};

// Loop Timing (180ms = ~5.5Hz, matching Python training timer)
const unsigned long CONTROL_LOOP_INTERVAL_MS = 180;
unsigned long last_step_time = 0;
static int step_count = 0;

// Episode State
static bool episode_ended = false;
static bool target_reached = false;
static int episode_count = 0;
static int success_count = 0;

// Collision Cooldown
unsigned long last_collision_time = 0;
const unsigned long COLLISION_COOLDOWN_MS = 1000;

void reset_episode() {
    reset_odometry();
    reset_lidar_state();  // Resets scans_since_reset and collision flag
    target_reached = false;
    episode_ended = false;
    step_count = 0;
    last_step_time = millis();
    last_collision_time = millis();  // Cooldown before first action
    episode_count++;
    logOut->println("================================================");
    logOut->print("Episode "); logOut->print(episode_count);
    logOut->println(" started. Robot is live.");
    logOut->print("Spawn: ("); logOut->print(SPAWN_X, 2);
    logOut->print(", "); logOut->print(SPAWN_Y, 2);
    logOut->print(") yaw="); logOut->println(SPAWN_YAW, 2);
    logOut->print("Target: ("); logOut->print(TARGET_X, 2);
    logOut->print(", "); logOut->print(TARGET_Y, 2); logOut->println(")");
    logOut->println("================================================");
}

void setup() {
    // 1. Kill motors immediately to prevent boot spin
    lockdown_motors();

    // 2. USB serial for minimal debug
    Serial.begin(115200);
    while (!Serial && millis() < 3000)
        ;

    // 3. Ethernet init
    Ethernet.begin(mac, ip);
    server.begin();

    Serial.println("[USB] Ethernet ready. Waiting for TCP client...");
    Serial.print("[USB] Connect to ");
    Serial.print(ip);
    Serial.print(":");
    Serial.println(ETH_PORT);
}

void loop() {
    switch (robotState) {

    // ================================================================
    // STATE 1: Wait for a TCP client to connect
    // ================================================================
    case WAIT_CLIENT: {
        ethClient = server.available();
        if (!ethClient) return;

        Serial.println("[USB] TCP client connected.");
        logOut = &ethClient;  // Switch all output to Ethernet

        logOut->println("========================================");
        logOut->println("DAGU RL Agent — Ethernet Control");
        logOut->println("Send any character to initialize and start...");
        logOut->println("========================================");

        robotState = WAIT_TRIGGER;
        break;
    }

    // ================================================================
    // STATE 2: Wait for any character from the TCP client to trigger
    // ================================================================
    case WAIT_TRIGGER: {
        if (!ethClient.connected()) {
            logOut = &Serial;
            Serial.println("[USB] Client disconnected while waiting.");
            robotState = WAIT_CLIENT;
            return;
        }
        if (!ethClient.available()) return;

        // Drain any extra bytes (e.g. CR+LF from terminal)
        while (ethClient.available()) ethClient.read();

        logOut->println();
        logOut->println("Trigger received! Initializing hardware...");
        logOut->println("========================================");

        robotState = INITIALIZING;
        break;
    }

    // ================================================================
    // STATE 3: Initialize LiDAR, IMU, encoders, then start episode
    // ================================================================
    case INITIALIZING: {
        // init_hardware();    // Starts Lidar serial and motor
        init_chassis();     // Starts Encoders, I2C IMU, and sets up PID gains

        logOut->println("Motor sanity test...");

        logOut->println("========================================");
        logOut->println("DAGU RL Agent Initialized. Starting episode.");
        logOut->println("========================================");

        reset_episode();
        robotState = RUNNING;
        break;
    }

    // ================================================================
    // STATE 4: Main RL control loop
    // ================================================================
    case RUNNING: {
        // Safety: if TCP client disconnects, stop motors
        if (!ethClient.connected()) {
            execute_motor_command(-1);
            update_chassis();
            logOut = &Serial;
            Serial.println("[USB] Client disconnected. Motors stopped.");
            robotState = WAIT_CLIENT;
            return;
        }

        // ====================================================================
        // 1. EPISODE ENDED — Wait for re-trigger
        // ====================================================================
        if (episode_ended) {
            // Continuously send STOP (locks PID at 0 AND feeds watchdog)
            execute_motor_command(-1);
            update_chassis();

            // Check for re-trigger: send any character to start new episode
            if (ethClient.available()) {
                while (ethClient.available()) ethClient.read();
                logOut->println();
                logOut->println("Re-trigger received! Starting new episode...");
                reset_episode();
            }
            return;
        }

        // ====================================================================
        // 2. BACKGROUND UPDATES (As fast as CPU allows)
        // ====================================================================
        // update_sensors();
        update_chassis();

        // ====================================================================
        // 3. RL INFERENCE (Strictly every 180ms)
        // ====================================================================
        unsigned long current_time = millis();
        if (current_time - last_step_time < CONTROL_LOOP_INTERVAL_MS) return;
        last_step_time = current_time;

        // --- COLLISION CHECK ---
        if (check_immediate_collision() || is_chassis_stalled()) {
            execute_motor_command(-1);
            logOut->println("EPISODE ENDED: Collision or Motor Stall.");
            logOut->println("Send any character to restart episode.");
            episode_ended = true;
            return;
        }

        // --- COLLISION COOLDOWN ---
        if (millis() - last_collision_time < COLLISION_COOLDOWN_MS) {
            execute_motor_command(-1);
            return;
        }

        // --- SUCCESS CHECK ---
        if (is_target_reached()) {
            execute_motor_command(-1);
            success_count++;
            logOut->print("EPISODE ENDED: SUCCESS! (");
            logOut->print(success_count); logOut->print("/");
            logOut->print(episode_count); logOut->println(")");
            logOut->println("Send any character to restart episode.");
            episode_ended = true;
            return;
        }

        // --- Q-TABLE LOOKUP ---
        int state_idx = get_state_index();

        int best_action = 0;
        float max_q_value = Q_TABLE[state_idx][0];
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (Q_TABLE[state_idx][a] > max_q_value) {
                max_q_value = Q_TABLE[state_idx][a];
                best_action = a;
            }
        }

        // --- EXECUTE ---
        execute_motor_command(best_action);
        step_count++;

        // --- LOGGING (gated by LOG_LEVEL in config.h) ---
#if LOG_LEVEL == 1
        // One compact line per step
        logOut->print("#"); logOut->print(step_count);
        logOut->print(" S:"); logOut->print(state_idx);
        logOut->print(" A:"); logOut->print(ACTION_NAMES[best_action]);
        logOut->print(" Q:"); logOut->print(max_q_value, 1);
        logOut->print(" | ");
        log_chassis_state(1);
        logOut->print(" | ");
        log_lidar_state(1);
        logOut->println();
#elif LOG_LEVEL >= 2
        // Multi-line detailed output per step
        logOut->println("--------------------------------------------");
        logOut->print("[RL]  step:"); logOut->print(step_count);
        logOut->print(" state:"); logOut->print(state_idx);
        logOut->print(" act:"); logOut->print(best_action);
        logOut->print("("); logOut->print(ACTION_NAMES[best_action]); logOut->print(")");
        logOut->print(" Q:"); logOut->print(max_q_value, 2);
        logOut->print(" [");
        for (int a = 0; a < NUM_ACTIONS; a++) {
            logOut->print(Q_TABLE[state_idx][a], 1);
            if (a < NUM_ACTIONS - 1) logOut->print(", ");
        }
        logOut->println("]");
        log_lidar_state(2);
        log_chassis_state(2);
#endif
        break;
    }
    }
}
