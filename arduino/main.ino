#include <Arduino.h>
#include "q_table.h"       // Your 2048x3 exported Q-table
#include "lidar_sensor.h"  // LiDAR API
#include "chassis.h"       // DAGU Dynamics API (Motors, Odom, Encoders)

/* ========================================================================
   RL AGENT CONFIGURATION
   ======================================================================== */
const int NUM_ACTIONS = 3;

// Loop Timing (180ms = ~5.5Hz, matching Python training timer)
const unsigned long CONTROL_LOOP_INTERVAL_MS = 180;
unsigned long last_step_time = 0;

void setup() {
    Serial.begin(115200);
    
    // Initialize hardware subsystems
    init_hardware();    // From lidar_sensor.h (starts Serial1 and Lidar motor)
    // init_chassis();  // (Placeholder) From teammate 2's chassis.h
    
    Serial.println("DAGU RL Agent Initialized.");
    Serial.println("Waiting for sensors to stabilize...");
    delay(2000); 
}

void loop() {
    // 1. BACKGROUND POLLING (Runs as fast as the CPU allows)
    // This constantly catches LiDAR bytes and updates the running minimums.
    // If an obstacle gets < 200mm, this function triggers emergency_stop() internally.
    update_sensors(); 

    // 2. RL INFERENCE STEP (Strictly every 180ms)
    unsigned long current_time = millis();
    if (current_time - last_step_time >= CONTROL_LOOP_INTERVAL_MS) {
        last_step_time = current_time;
        
        // --- TERMINAL CONDITION CHECKS ---
        
        // Check if the LiDAR reflex system triggered an emergency stop
        if (check_immediate_collision()) {
            Serial.println("STATUS: COLLISION PREVENTED. Waiting for clear path.");
            // The motors are already stopped by lidar_sensor.cpp.
            // We return early to prevent the RL agent from sending a new drive command.
            return;
        }

        // Check if we reached the target (Placeholder function for teammate 2)
        if (is_target_reached()) {
            execute_motor_command(-1); // -1 = Stop command
            Serial.println("STATUS: SUCCESS! Target Reached.");
            return; 
        }

        // --- FETCH STATE ---
        // All the heavy math (distance zones, sector matching, bit-shifting) 
        // is hidden cleanly behind this single function call.
        int state_idx = get_state_index();

        // --- Q-TABLE LOOKUP (ARGMAX) ---
        int best_action = 0;
        float max_q_value = Q_TABLE[state_idx];
        
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (Q_TABLE[state_idx][a] > max_q_value) {
                max_q_value = Q_TABLE[state_idx][a];
                best_action = a;
            }
        }

        // --- EXECUTE ACTION ---
        // 0: Forward, 1: Left, 2: Right
        execute_motor_command(best_action);
        
        // --- DEBUGGING ---
        // Serial.print("State Index: "); Serial.print(state_idx);
        // Serial.print(" | Action: "); Serial.println(best_action);
    }
}