/*
 * TEST: IMU (BNO055)
 * ===================
 * Verifies BNO055 detection, gyro calibration, Z-axis sign convention,
 * and heading integration. Uses EXACT same setup as main firmware.
 *
 * HOW TO USE:
 *   1. Upload to Portenta H7
 *   2. Open Serial Monitor at 115200
 *   3. Keep robot STILL during calibration (wait for "Calibrated!")
 *   4. Rotate robot LEFT (CCW from above)  → gyroZ should be POSITIVE
 *   5. Rotate robot RIGHT (CW from above)  → gyroZ should be NEGATIVE
 *   6. Heading should increase when turning left, decrease when turning right
 *
 * IF GYRO Z SIGN IS WRONG:
 *   In chassis.cpp, change readOmegaZ_IMU() to return -g.z()
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

// ---- BNO055 Setup (SAME as chassis.cpp) ----
Adafruit_BNO055 bno(55, 0x28);

// ---- Heading Integration (SAME logic as chassis.cpp) ----
float yaw_rad = 0.0f;

static inline float wrapPi(float a)
{
    const float PI_VAL = 3.1415926f;
    while (a > PI_VAL)  a -= 2.0f * PI_VAL;
    while (a < -PI_VAL) a += 2.0f * PI_VAL;
    return a;
}

void setup()
{
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    Wire.begin();
    if (!bno.begin())
    {
        Serial.println("ERROR: BNO055 not detected! Check wiring (I2C, addr 0x28).");
        while (1) delay(10);
    }
    bno.setExtCrystalUse(true);

    Serial.println("================================================");
    Serial.println("IMU TEST — Calibrating gyroscope...");
    Serial.println("DO NOT MOVE the robot during calibration.");
    Serial.println("================================================");

    // Same calibration loop as chassis.cpp
    uint8_t system, gyro, accel, mag;
    system = gyro = accel = mag = 0;

    while (gyro < 3)
    {
        bno.getCalibration(&system, &gyro, &accel, &mag);
        Serial.print("Cal — Sys:");
        Serial.print(system);
        Serial.print(" Gyro:");
        Serial.print(gyro);
        Serial.print(" Accel:");
        Serial.print(accel);
        Serial.print(" Mag:");
        Serial.println(mag);
        delay(200);
    }

    Serial.println("================================================");
    Serial.println("Gyro CALIBRATED!");
    Serial.println("Now rotate the robot:");
    Serial.println("  LEFT (CCW from above)  → gyroZ should be POSITIVE");
    Serial.println("  RIGHT (CW from above)  → gyroZ should be NEGATIVE");
    Serial.println("================================================");

    yaw_rad = 0.0f;
}

void loop()
{
    static uint32_t lastMs = 0;
    uint32_t now = millis();
    if (now - lastMs < 100) return; // Print every 100ms

    float dt = (float)(now - lastMs) / 1000.0f;
    lastMs = now;

    // Read gyro Z (SAME as readOmegaZ_IMU in chassis.cpp)
    imu::Vector<3> g = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    float gyroZ = g.z();

    // Integrate heading (SAME as update_chassis)
    yaw_rad = wrapPi(yaw_rad + gyroZ * dt);
    float yaw_deg = yaw_rad * 180.0f / 3.1415926f;

    // Also read Euler heading for cross-reference
    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

    Serial.print("gyroZ: ");
    Serial.print(gyroZ, 3);
    Serial.print(" rad/s | yaw: ");
    Serial.print(yaw_rad, 3);
    Serial.print(" rad (");
    Serial.print(yaw_deg, 1);
    Serial.print(" deg) | euler_heading: ");
    Serial.println(euler.x(), 1);
}
