#include "lidar_sensor.h"
#include "chassis.h" // Brings in emergency_stop() and get_current_pose()
#include "config.h"
#include "mbed.h"
#include <RPLidar.h>
#include <Arduino.h>
#include <math.h>

#define BAUD_RATE 115200
#define RPLIDAR_MOTOR PJ_7

RPLidar lidar;

// Internal Lidar State Variables
int lidar_state[5] = {0, 0, 0, 0, 0};
float sector_min_distance[5];
float lidar_target_angle = 0.0;
float target_angle_ccw_deg = 0.0;   // CCW convention for Q-table sector 
float distance_to_target = 0.0;
int target_vis = 0;
float min_dist_in_target_window = 12000.0;
bool collision_detected = false;

// Thresholds
const float SAFE_DIST = 600.0;
const float COLLISION_DIST = 200.0;
const float RANGE_MIN = 150.0;
const float RANGE_MAX = 12000.0;

const int NUM_DISTANCE_ZONES = 4;
const int NUM_TARGET_SECTORS = 8;

const float SECTORS[5][2] = {
    {270.0, 306.0}, // 0: Far Left
    {306.0, 342.0}, // 1: Left
    {342.0, 18.0},  // 2: Front
    {18.0, 54.0},   // 3: Right
    {54.0, 90.0}    // 4: Far Right
};

int scans_since_reset = 0;

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

void init_hardware()
{
  Serial1.begin(BAUD_RATE);
  lidar.begin(Serial1);

  // Start motor once and begin scanning
  analogWrite(RPLIDAR_MOTOR, 255);
  delay(1000);                 // Wait for motor to reach scan speed
  lidar.startScan();

  reset_sector_minimums();
}

void update_sensors()
{
  if (IS_OK(lidar.waitPoint()))
  {
    float distance = lidar.getCurrentPoint().distance;
    float angle = lidar.getCurrentPoint().angle;
    bool startBit = lidar.getCurrentPoint().startBit;

    // If 360 spin is complete, finalize states
    if (startBit)
    {
      update_lidar_state();
      is_target_visible();

      reset_sector_minimums();
      min_dist_in_target_window = RANGE_MAX;

      calculate_target_angle();

      scans_since_reset++;
    }

    // Clean Data
    if (distance == 0.0 || distance > RANGE_MAX)
    {
      distance = RANGE_MAX;
    }
    else if (distance < RANGE_MIN)
    {
      distance = RANGE_MAX;
    }

    // Update running minimums
    find_target_ray(distance, angle);
    update_sector_min_distance(distance, angle);

    // Check for immediate emergency stop
    check_collision(distance);
  }
  else
  {
    // Transient timeout — do nothing, library resyncs naturally.
    // Motor keeps spinning, scan keeps running.
  }
}

bool check_immediate_collision()
{
  bool current_col = collision_detected;
  collision_detected = false; // Reset flag after reading by the main loop
  return current_col;
}

int get_state_index()
{
  // 1. Convert the 5-bit array into a single integer
  int lidar_int = 0;
  for (int i = 0; i < 5; i++)
  {
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
  if (scans_since_reset < 3) return;
  if (distance < COLLISION_DIST) {
    collision_detected = true;
    emergency_stop();
  }
}

void calculate_target_angle()
{
  float current_x, current_y, current_yaw;
  get_current_pose(current_x, current_y, current_yaw);

  float dx = TARGET_X - current_x;
  float dy = TARGET_Y - current_y;
  distance_to_target = sqrt((dx * dx) + (dy * dy));

  float global_target_angle = atan2(dy, dx);
  float relative_angle = global_target_angle - current_yaw;
  float theta_robot_deg = relative_angle * (180.0 / PI);

  // --- Store CCW angle for target sector (Python convention) ---
  float ccw_deg = theta_robot_deg;
  while (ccw_deg > 180.0)  ccw_deg -= 360.0;
  while (ccw_deg < -180.0) ccw_deg += 360.0;
  target_angle_ccw_deg = ccw_deg;  // kept in [-180, 180] like Python

  // --- CW conversion for RPLIDAR ray matching only ---
  theta_robot_deg = 360.0 - theta_robot_deg;
  while (theta_robot_deg >= 360.0) theta_robot_deg -= 360.0;
  while (theta_robot_deg < 0.0)    theta_robot_deg += 360.0;
  lidar_target_angle = theta_robot_deg;
}

void reset_sector_minimums()
{
  for (int i = 0; i < 5; i++)
  {
    sector_min_distance[i] = RANGE_MAX;
  }
}

void update_sector_min_distance(float distance, float angle)
{
  for (int i = 0; i < 5; i++)
  {
    float start_angle = SECTORS[i][0];
    float end_angle = SECTORS[i][1];

    bool inSector = false;
    if (start_angle < end_angle)
    {
      inSector = (angle >= start_angle && angle <= end_angle);
    }
    else
    {
      inSector = (angle >= start_angle || angle <= end_angle); // Wraparound for Front
    }

    if (inSector)
    {
      if (distance < sector_min_distance[i])
      {
        sector_min_distance[i] = distance;
      }
      break;
    }
  }
}

void update_lidar_state()
{
  for (int i = 0; i < 5; i++)
  {
    lidar_state[i] = (sector_min_distance[i] > SAFE_DIST);
  }
}

void find_target_ray(float distance, float angle)
{
  float angle_diff = abs(angle - lidar_target_angle);

  if (angle_diff > 180.0)
  {
    angle_diff = 360.0 - angle_diff;
  }

  if (angle_diff <= 3.0)
  {
    if (distance < min_dist_in_target_window)
    {
      min_dist_in_target_window = distance;
    }
  }
}

void is_target_visible()
{
  float dist_mm = distance_to_target * 1000.0f; // convert to mm
  target_vis = (min_dist_in_target_window > dist_mm) ? 1 : 0;
}

int get_distance_zone()
{
  if (distance_to_target < 1.0f)
    return 0;
  if (distance_to_target < 2.0f)
    return 1;
  if (distance_to_target < 3.0f)
    return 2;
  return 3;
}

int get_target_sector()
{
  float deg = target_angle_ccw_deg;  // [-180, 180]
  if (deg < 0) deg += 360.0;        // [0, 360) — same as Python
  float shifted = deg + 22.5;
  if (shifted >= 360.0) shifted -= 360.0;
  return ((int)(shifted / 45.0)) % 8;
}

void reset_lidar_state() {
    scans_since_reset = 0;
    collision_detected = false;
    min_dist_in_target_window = RANGE_MAX;
    reset_sector_minimums();
    for (int i = 0; i < 5; i++) lidar_state[i] = 0;
    target_vis = 0;
}

void log_lidar_state(int level)
{
    if (level <= 0) return;

    int distance_zone = get_distance_zone();
    int target_sector = get_target_sector();

    if (level == 1)
    {
        // Compact: L:10110 vis:1 dz:2 ts:3 dist:2.35
        logOut->print("L:");
        for (int i = 0; i < 5; i++) logOut->print(lidar_state[i]);
        logOut->print(" vis:"); logOut->print(target_vis);
        logOut->print(" dz:");  logOut->print(distance_zone);
        logOut->print(" ts:");  logOut->print(target_sector);
        logOut->print(" d:");   logOut->print(distance_to_target, 2);
    }
    else
    {
        // Detailed multi-line
        logOut->print("[LDR] sectors: ");
        for (int i = 0; i < 5; i++) {
            logOut->print(lidar_state[i]);
            logOut->print(" ");
        }
        logOut->print("| mins(mm): ");
        for (int i = 0; i < 5; i++) {
            logOut->print((int)sector_min_distance[i]);
            logOut->print(" ");
        }
        logOut->println();

        logOut->print("[LDR] vis:"); logOut->print(target_vis);
        logOut->print(" dz:");  logOut->print(distance_zone);
        logOut->print(" ts:");  logOut->print(target_sector);
        logOut->print(" | dist:"); logOut->print(distance_to_target, 3);
        logOut->print("m | tgt_ccw:"); logOut->print(target_angle_ccw_deg, 1);
        logOut->print(" tgt_cw:"); logOut->print(lidar_target_angle, 1);
        logOut->print(" | scans:"); logOut->println(scans_since_reset);
    }
}