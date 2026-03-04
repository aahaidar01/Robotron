#ifndef LIDAR_SENSOR_H
#define LIDAR_SENSOR_H

/* ========================================================================
   EXPOSED API FUNCTIONS
   These are the only functions the main.ino RL loop is allowed to call.
   All the messy math is hidden safely inside lidar_sensor.cpp.
   ======================================================================== */

// 1. Hardware Initialization
// Starts the Lidar serial connection and motor
void init_hardware();

// 2. Continuous Polling
// Must be called as fast as possible in the main loop() to catch Lidar bytes
void update_sensors();

// 3. Safety Check
// Returns true if an obstacle is dangerously close (< 200mm)
// Note: emergency_stop() is already triggered internally, but this lets the main loop know.
bool check_immediate_collision();

// 4. State Calculation
// Runs all the target and distance math, returning the final 0-2047 Q-table index
int get_state_index();

void reset_lidar_state();

// 5. Diagnostics — call from main loop, gated by LOG_LEVEL
//    level 1: compact one-line summary (sectors + target info)
//    level 2: full detail (sector distances, angles, scans count)
void log_lidar_state(int level);

#endif // LIDAR_SENSOR_H