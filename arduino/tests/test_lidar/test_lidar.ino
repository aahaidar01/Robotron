/*
 * TEST: LiDAR (RPLIDAR A1)
 * =========================
 * Verifies RPLIDAR motor spin, scan data reception, and sector classification.
 * Uses EXACT same pin, sector definitions, and processing logic as main firmware.
 *
 * HOW TO USE:
 *   1. Upload to Portenta H7
 *   2. Open Serial Monitor at 115200
 *   3. LiDAR motor should spin — scan count should increment at ~5.5Hz
 *   4. Place hand in FRONT of robot  → sector 2 (Front) distance drops, state=0
 *   5. Place hand to LEFT of robot   → sector 0 (Far Left) distance drops, state=0
 *   6. Place hand to RIGHT of robot  → sector 4 (Far Right) distance drops, state=0
 *
 * IF LIDAR DOESN'T SPIN:
 *   Check PJ_7 wiring, check power supply to RPLIDAR motor
 *
 * IF WRONG SECTOR RESPONDS:
 *   Check robot orientation vs RPLIDAR mounting direction
 *   RPLIDAR 0° should face robot's forward direction
 */

#include "mbed.h"
#include <RPLidar.h>
#include <Arduino.h>
#include <math.h>

// ---- RPLIDAR Setup (SAME as lidar_sensor.cpp) ----
#define BAUD_RATE 115200
#define RPLIDAR_MOTOR PJ_7

RPLidar lidar;

// ---- Thresholds (SAME as lidar_sensor.cpp) ----
const float SAFE_DIST = 600.0;
const float RANGE_MIN = 150.0;
const float RANGE_MAX = 12000.0;

// ---- Sector Definitions (SAME as lidar_sensor.cpp) ----
// RPLIDAR uses CW convention: 0°=front, 90°=right, 270°=left
const char* SECTOR_NAMES[] = {"FarLeft", "Left", "Front", "Right", "FarRight"};
const float SECTORS[5][2] = {
    {270.0, 306.0}, // 0: Far Left
    {306.0, 342.0}, // 1: Left
    {342.0, 18.0},  // 2: Front (wraparound)
    {18.0, 54.0},   // 3: Right
    {54.0, 90.0}    // 4: Far Right
};

// ---- State Variables ----
float sector_min_distance[5];
int lidar_state[5] = {0, 0, 0, 0, 0};
int scan_count = 0;

void reset_sector_minimums()
{
    for (int i = 0; i < 5; i++)
        sector_min_distance[i] = RANGE_MAX;
}

// SAME logic as lidar_sensor.cpp (with the fixed [0]/[1] indexing)
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
            inSector = (angle >= start_angle || angle <= end_angle);
        }

        if (inSector)
        {
            if (distance < sector_min_distance[i])
                sector_min_distance[i] = distance;
            break;
        }
    }
}

void update_lidar_state()
{
    for (int i = 0; i < 5; i++)
        lidar_state[i] = (sector_min_distance[i] > SAFE_DIST);
}

void print_scan_results()
{
    scan_count++;

    Serial.print("Scan #");
    Serial.print(scan_count);
    Serial.println(":");

    // Sector distances
    Serial.print("  Distances(mm): ");
    for (int i = 0; i < 5; i++)
    {
        Serial.print(SECTOR_NAMES[i]);
        Serial.print("=");
        Serial.print((int)sector_min_distance[i]);
        Serial.print("  ");
    }
    Serial.println();

    // Binary states
    Serial.print("  States (1=free, 0=blocked): ");
    for (int i = 0; i < 5; i++)
    {
        Serial.print(SECTOR_NAMES[i]);
        Serial.print("=");
        Serial.print(lidar_state[i]);
        Serial.print("  ");
    }
    Serial.println();
    Serial.println();
}

void setup()
{
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    Serial1.begin(BAUD_RATE);
    lidar.begin(Serial1);
    pinMode(RPLIDAR_MOTOR, OUTPUT);

    Serial.println("================================================");
    Serial.println("LIDAR TEST — Place obstacles around the robot");
    Serial.println("Sector layout (top-down, CW from front):");
    Serial.println("  FarLeft(270-306) Left(306-342) Front(342-18)");
    Serial.println("  Right(18-54) FarRight(54-90)");
    Serial.println("SAFE_DIST = 600mm — closer = blocked (state=0)");
    Serial.println("================================================");

    reset_sector_minimums();
}

void loop()
{
    if (IS_OK(lidar.waitPoint()))
    {
        float distance = lidar.getCurrentPoint().distance;
        float angle = lidar.getCurrentPoint().angle;
        bool startBit = lidar.getCurrentPoint().startBit;

        // Full 360° rotation complete
        if (startBit)
        {
            update_lidar_state();
            print_scan_results();
            reset_sector_minimums();
        }

        // Clean data (SAME as lidar_sensor.cpp)
        if (distance == 0.0 || distance > RANGE_MAX)
            distance = RANGE_MAX;
        else if (distance < RANGE_MIN)
            distance = RANGE_MAX;

        update_sector_min_distance(distance, angle);
    }
    else
    {
        // Restart LiDAR if connection drops
        analogWrite(RPLIDAR_MOTOR, 0);
        rplidar_response_device_info_t info;
        if (IS_OK(lidar.getDeviceInfo(info, 100)))
        {
            Serial.println("LiDAR connected — starting scan...");
            lidar.startScan();
            analogWrite(RPLIDAR_MOTOR, 255);
        }
        delay(500);
    }
}
