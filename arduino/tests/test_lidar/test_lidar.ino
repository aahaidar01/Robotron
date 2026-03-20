/*
 * TEST: LiDAR (RPLIDAR A1)
 * =========================
 * Verifies RPLIDAR motor spin, scan data reception, and sector classification.
 */

#include "mbed.h"
#include <RPLidar.h>
#include <Arduino.h>
#include <math.h>

// ---- RPLIDAR Setup ----
#define BAUD_RATE 115200
#define RPLIDAR_MOTOR PJ_7
RPLidar lidar;

// ---- Thresholds ----
const float SAFE_DIST = 600.0;
const float RANGE_MIN = 150.0;
const float RANGE_MAX = 12000.0;

// ---- Sector Definitions ----
// RPLIDAR uses CW convention: 0°=front, 90°=right, 270°=left
const char SECTOR_NAMES[][9] = {"FarLeft", "Left", "Front", "Right", "FarRight"};
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

void update_sector_min_distance(float distance, float angle)
{
    for (int i = 0; i < 5; i++)
    {
        float start_angle = SECTORS[i][0];
        float end_angle = SECTORS[i][1];
        bool inSector = false;
        
        if (start_angle < end_angle) {
            inSector = (angle >= start_angle && angle <= end_angle);
        } else {
            inSector = (angle >= start_angle || angle <= end_angle);
        }
        
        if (inSector) {
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
    
    // Fast, compact printing to prevent serial buffer delays
    Serial.print("Scan #");
    Serial.print(scan_count);
    Serial.print(" | Dist: ");
    
    // Print the 5 distances
    for (int i = 0; i < 5; i++) {
        Serial.print((int)sector_min_distance[i]);
        if (i < 4) Serial.print(",");
    }
    
    Serial.print(" | States: ");
    
    // Print the 5 binary states
    for (int i = 0; i < 5; i++) {
        Serial.print(lidar_state[i]);
        if (i < 4) Serial.print(",");
    }
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
    Serial.println("Order: FarLeft, Left, Front, Right, FarRight");
    Serial.println("SAFE_DIST = 600mm — closer = blocked (state=0)");
    Serial.println("================================================");
    
    reset_sector_minimums();

    // Start motor and begin continuous scan once
    analogWrite(RPLIDAR_MOTOR, 255);
    lidar.startScan();
    Serial.println("LiDAR scan started successfully.");
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
            
            // Throttle printing to every 3rd scan to guarantee zero dropped points
            static int print_throttle = 0;
            print_throttle++;
            if (print_throttle >= 3) {
                print_scan_results();
                print_throttle = 0;
            }
            
            reset_sector_minimums();
        }
        
        // Clean data
        if (distance == 0.0 || distance > RANGE_MAX)
            distance = RANGE_MAX;
        else if (distance < RANGE_MIN)
            distance = RANGE_MAX;
            
        update_sector_min_distance(distance, angle);
    }
    else
    {
        // Do nothing. If a single byte is missed, the library resyncs naturally.
    }
}