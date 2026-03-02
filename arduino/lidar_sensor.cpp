#include "lidar_sensor.h"
#include "chassis.h" // Brings in emergency_stop() and get_current_pose()
#include <RPLidar.h>
#include <Arduino.h>
#include <math.h>

#define BAUD_RATE 115200
#define RPLIDAR_MOTOR 3

RPLidar lidar;

// Internal Lidar State Variables
int lidar_state = {0, 0, 0, 0, 0};
float sector_min_distance;
float lidar_target_angle = 0.0; 
float distance_to_target = 0.0;
int target_vis = 0; 
float min_dist_in_target_window = 12000.0;
bool collision_detected = false; 

// Thresholds
const float SAFE_DIST = 400.0;
const float COLLISION_DIST = 200.0;
const float RANGE_MIN = 150.0;
const float RANGE_MAX = 12000.0;

const int NUM_DISTANCE_ZONES = 4;
const int NUM_TARGET_SECTORS = 8;
const float TARGET_X  = 0.0;
const float TARGET_Y = 1800.0; // Assuming mm convention

const float SECTORS = {
  {270.0, 306.0},   // 0: Far Left
  {306.0, 342.0},   // 1: Left
  {342.0, 18.0},    // 2: Front
  {18.0, 54.0},     // 3: Right
  {54.0, 90.0}      // 4: Far Right
};

// --- Forward Declarations ---
void reset_sector_minimums();
void update_sector_min_distance(float distance, float angle);
void check_collision(float distance);
void update_lidar_state();
void calculate_target_angle();
void find_target_ray(float distance, float angle);
void is_target_visible();
int get_distance_zone();
int get_target_sector();

// =======================================================
// EXPOSED API FUNCTIONS (Called by the main .ino script)
// =======================================================

void init_hardware() {
  Serial1.begin(BAUD_RATE);
  lidar.begin(Serial1);
  pinMode(RPLIDAR_MOTOR, OUTPUT);
  reset_sector_minimums();
}

void update_sensors() {
  if (IS_OK(lidar.waitPoint())) {
    float distance = lidar.getCurrentPoint().distance; 
    float angle    = lidar.getCurrentPoint().angle; 
    bool  startBit = lidar.getCurrentPoint().startBit; 
    
    // If 360 spin is complete, finalize states
    if (startBit) { 
      update_lidar_state();
      is_target_visible();
      
      reset_sector_minimums();
      min_dist_in_target_window = RANGE_MAX;
      
      calculate_target_angle(); 
    }

    // Clean Data 
    if (distance == 0.0 || distance > RANGE_MAX){
      distance = RANGE_MAX;
    } else if (distance < RANGE_MIN) {
      distance = RANGE_MAX; 
    }

    // Update running minimums
    find_target_ray(distance, angle); 
    update_sector_min_distance(distance, angle);
    
    // Check for immediate emergency stop
    check_collision(distance);

  } else {
    // Attempt to restart Lidar if connection drops
    analogWrite(RPLIDAR_MOTOR, 0); 
    rplidar_response_device_info_t info;
    if (IS_OK(lidar.getDeviceInfo(info, 100))) {
       lidar.startScan();
       analogWrite(RPLIDAR_MOTOR, 255);
    }
  }
}

bool check_immediate_collision() {
    bool current_col = collision_detected;
    collision_detected = false; // Reset flag after reading by the main loop
    return current_col;
}

int get_state_index() {
  // 1. Convert the 5-bit array into a single integer
  int lidar_int = 0;
  for (int i = 0; i < 5; i++) {
    lidar_int += (lidar_state[i] << i); 
  }

  // 2. Get the other variables
  int distance_zone = get_distance_zone();
  int target_sector = get_target_sector();

  // 3. Multiply them together exactly like the Python code
  int state_idx = (lidar_int * (2 * NUM_DISTANCE_ZONES * NUM_TARGET_SECTORS)) +
                  (target_vis * (NUM_DISTANCE_ZONES * NUM_TARGET_SECTORS)) +
                  (distance_zone * NUM_TARGET_SECTORS) +
                  target_sector;

  return state_idx;
}

// =======================================================
// INTERNAL HELPER FUNCTIONS 
// =======================================================

void check_collision(float distance) {
  if (distance < COLLISION_DIST) {
    collision_detected = true; 
    // INSTANT REFLEX: Bypass RL timer and kill motors immediately
    emergency_stop(); 
  }
}

void calculate_target_angle() {
  // 1. Fetch the latest odometry directly from the chassis teammate's code
  float current_x, current_y, current_yaw;
  get_current_pose(current_x, current_y, current_yaw);

  // 2. Do the math using the local variables
  float dx = TARGET_X - current_x;
  float dy = TARGET_Y - current_y;
  distance_to_target = sqrt((dx*dx) + (dy*dy)) * 1000.0;
  
  float global_target_angle = atan2(dy, dx); 
  float relative_angle = global_target_angle - current_yaw;
  float theta_robot_deg = relative_angle * (180.0 / PI);
  
  theta_robot_deg = 360.0 - theta_robot_deg; // CCW to CW fix
  
  while (theta_robot_deg >= 360.0) theta_robot_deg -= 360.0;
  while (theta_robot_deg < 0.0) theta_robot_deg += 360.0;
  
  lidar_target_angle = theta_robot_deg;
}

void reset_sector_minimums() {
  for (int i = 0; i < 5; i++) {
    sector_min_distance[i] = RANGE_MAX;
  }
}

void update_sector_min_distance(float distance, float angle) {
  for (int i=0; i<5; i++) {
    float start_angle = SECTORS[i];
    float end_angle = SECTORS[i];

    bool inSector = false;
    if (start_angle < end_angle){
      inSector = (angle >= start_angle && angle <= end_angle);
    } else {
      inSector = (angle >= start_angle || angle <= end_angle); // Wraparound for Front
    }

    if (inSector) {
      if (distance < sector_min_distance[i]) {
        sector_min_distance[i] = distance;
      }
      break; 
    }
  }
}

void update_lidar_state() {
  for (int i=0; i<5; i++) {
    lidar_state[i] = (sector_min_distance[i] > SAFE_DIST);
  }
}

void find_target_ray(float distance, float angle) {
  float angle_diff = abs(angle - lidar_target_angle);

  if (angle_diff > 180.0) {
    angle_diff = 360.0 - angle_diff; 
  }

  if (angle_diff <= 3.0) {
    if (distance < min_dist_in_target_window) {
      min_dist_in_target_window = distance;
    }
  }
}

void is_target_visible() {
  if (min_dist_in_target_window > distance_to_target) {
    target_vis = 1;
  } else {
    target_vis = 0;
  }
}

int get_distance_zone() {
  if (distance_to_target < 1000.0) return 0;
  if (distance_to_target < 2000.0) return 1;
  if (distance_to_target < 3000.0) return 2;
  return 3; 
}

int get_target_sector() {
  float shifted_angle = lidar_target_angle + 22.5;
  if (shifted_angle >= 360.0) shifted_angle -= 360.0;
  
  int sector = (int)(shifted_angle / 45.0);
  return sector % 8; 
}