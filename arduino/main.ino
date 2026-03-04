#include <Arduino.h>
#include "config.h"        // Shared constants (TARGET_X, TARGET_Y, etc.)
#include "q_table.h"       // Your 2048x3 exported Q-table
#include "lidar_sensor.h"  // LiDAR, Target Math, Reflexes
#include "chassis.h"       // Motors, Odom, Sensor Fusion


const int RESET_BUTTON_PIN = 10;

/* ========================================================================
   RL AGENT CONFIGURATION
   ======================================================================== */
const int NUM_ACTIONS = 3;

// Loop Timing (180ms = ~5.5Hz, matching Python training timer)
const unsigned long CONTROL_LOOP_INTERVAL_MS = 180;
unsigned long last_step_time = 0;

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
    last_step_time = millis();
    last_collision_time = millis();  // Cooldown before first action
    episode_count++;
    Serial.println("================================================");
    Serial.print("Episode "); Serial.print(episode_count);
    Serial.println(" started. Robot is live.");
    Serial.print("Spawn: ("); Serial.print(SPAWN_X, 2);
    Serial.print(", "); Serial.print(SPAWN_Y, 2);
    Serial.print(") yaw="); Serial.println(SPAWN_YAW, 2);
    Serial.print("Target: ("); Serial.print(TARGET_X, 2);
    Serial.print(", "); Serial.print(TARGET_Y, 2); Serial.println(")");
    Serial.println("================================================");
}

void setup() {
    Serial.begin(115200);
    
    // Reset button
    pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);
    
    // Initialize hardware subsystems
    init_hardware();    // Starts Lidar serial and motor
    init_chassis();     // Starts Encoders, I2C IMU, and sets up PID gains
    
    Serial.println("================================================");
    Serial.println("DAGU RL Agent Initialized.");
    Serial.println("Press button to start first episode.");
    Serial.println("================================================");
    
    // Wait for button press before first episode
    episode_ended = true;
}

void loop() {
    // ====================================================================
    // 1. EPISODE ENDED — Wait for manual reset
    // ====================================================================
    if (episode_ended) {
        // Keep chassis updated so PID stays at zero cleanly
        update_chassis();
        
        if (digitalRead(RESET_BUTTON_PIN) == LOW) {
            delay(50);  // Debounce
            if (digitalRead(RESET_BUTTON_PIN) == LOW) {  // Confirm still pressed
                reset_episode();
            }
        }
        return;
    }

    // ====================================================================
    // 2. BACKGROUND UPDATES (As fast as CPU allows)
    // ====================================================================
    update_sensors();
    update_chassis();

    // ====================================================================
    // 3. RL INFERENCE (Strictly every 180ms)
    // ====================================================================
    unsigned long current_time = millis();
    if (current_time - last_step_time < CONTROL_LOOP_INTERVAL_MS) return;
    last_step_time = current_time;

    // --- COLLISION CHECK ---
    if (check_immediate_collision()) {
        execute_motor_command(-1);
        Serial.println("EPISODE ENDED: Collision.");
        Serial.println("Place robot at start. Press button to restart.");
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
        Serial.print("EPISODE ENDED: SUCCESS! (");
        Serial.print(success_count); Serial.print("/");
        Serial.print(episode_count); Serial.println(")");
        Serial.println("Place robot at start. Press button to restart.");
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
}