#include "chassis.h"
#include "config.h"
#include "mbed.h"
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

// ================== PINS & CONFIG ==================

// ---- Encoder Pin Definitions ----
// Hat Carrier 40-pin header mapping:
//   GPIO2 (pin 11) -> PD_4   -> Encoder 1 Channel A
//   GPIO6 (pin 13) -> PG_10  -> Encoder 1 Channel B
//   PWM2  (pin 26) -> Arduino 4 (PC_7)   -> Encoder 2 Channel A
//   PWM6  (pin 37) -> Arduino 0 (PH_15)  -> Encoder 2 Channel B
//
// NOTE: GPIO2 and GPIO6 don't have simple Arduino pin numbers,
//       so we use mbed pin names for those.

const int ENC_L_A = PD_4;
const int ENC_L_B = PG_10;
const int ENC_R_A = 4;
const int ENC_R_B = 0;

// ---- Motor Pin Definitions (Arduino pin numbers) ----
// Hat Carrier 40-pin header mapping:
//   PWM0 (pin 7)  -> Arduino 6  (PA_8)  -> Motor 1 PWM
//   PWM1 (pin 22) -> Arduino 5  (PC_6)  -> Motor 1 DIR
//   PWM4 (pin 33) -> Arduino 2  (PJ_11) -> Motor 2 PWM
//   PWM5 (pin 36) -> Arduino 1  (PK_1)  -> Motor 2 DIR

const int PWM_L = 6;
const int DIR_L = 5;
const int PWM_R = 2;
const int DIR_R = 1;


static constexpr int PWM_LIMIT = 191;
static constexpr uint32_t CTRL_DT_MS = 20; // 50Hz PID loop

// Geometry & Scaling
static constexpr float TRACK_WIDTH_M = 0.26f;
static constexpr float CPR_MOTOR = 48.0f;
static constexpr float GEAR_RATIO_EXACT = 34.014f;
static constexpr float CPR_WHEEL = CPR_MOTOR * GEAR_RATIO_EXACT;
static constexpr float WHEEL_DIAM_M = 0.12f;
static constexpr float WHEEL_CIRC_M = 3.1415926f * WHEEL_DIAM_M;
static constexpr float METERS_PER_COUNT = WHEEL_CIRC_M / CPR_WHEEL;

// Fusion Weight
static constexpr float ALPHA = 0.85f;

// RL Agent Target Speeds
float current_vTarget = 0.0f;
float current_omegaTarget = 0.0f;

// Base Speeds
float V_FWD = 0.18f;
float V_TURN = 0.03f;
static constexpr float OMEGA_TURN = 0.5f;

// --- Global Odometry Variables ---
static float odom_x_m = 0.0f;
static float odom_y_m = 0.0f;
static float odom_th_rad = 0.0f;
static float yaw_offset_rad = 0.0f;

// --- Safety Variables ---
static uint32_t stall_timer_ms = 0;
const uint32_t STALL_TIMEOUT_MS = 800;
static uint32_t last_command_time_ms = 0;

// IMU
Adafruit_BNO055 bno(55, 0x28);

// ================== PID CONTROLLER ==================
struct PID
{
    float Kp = 0, Ki = 0, Kd = 0;
    float integ = 0, prevErr = 0;
    float outMin = -PWM_LIMIT, outMax = PWM_LIMIT;

    void reset()
    {
        integ = 0.0f;
        prevErr = 0.0f;
    }

    float update(float err, float dt)
    {
        // Zero-crossing reset to prevent post-obstruction lunge
        if ((err > 0 && prevErr < 0) || (err < 0 && prevErr > 0))
        {
            integ = 0.0f;
        }

        integ += err * dt;
        if (Ki > 1e-6f)
        {
            float iMax = outMax / Ki;
            float iMin = outMin / Ki;
            if (integ > iMax)
                integ = iMax;
            if (integ < iMin)
                integ = iMin;
        }
        float deriv = (err - prevErr) / dt;
        prevErr = err;
        float u = Kp * err + Ki * integ + Kd * deriv;
        if (u > outMax)
            u = outMax;
        if (u < outMin)
            u = outMin;
        return u;
    }
};

PID pidSpeed;
PID pidOmega;

// ================== ENCODER ISRs ==================
volatile int32_t ticksL = 0;
volatile int32_t ticksR = 0;
int32_t prevTicksL = 0, prevTicksR = 0;

void isrEncL_A()
{
    bool a = digitalRead(ENC_L_A);
    bool b = digitalRead(ENC_L_B);
    ticksL += (a == b) ? +1 : -1;
}

void isrEncR_A()
{
    bool a = digitalRead(ENC_R_A);
    bool b = digitalRead(ENC_R_B);
    ticksR += (a == b) ? +1 : -1;
}

// ================== HELPER MATH ==================
static inline float wrapPi(float a)
{
    const float PI_VAL = 3.1415926f;
    while (a > PI_VAL)
        a -= 2.0f * PI_VAL;
    while (a < -PI_VAL)
        a += 2.0f * PI_VAL;
    return a;
}

static inline int clampMag(int mag) { return (mag > PWM_LIMIT) ? PWM_LIMIT : mag; }

void setMotorPWMDIR(int cmdL, int cmdR)
{
    int magL = clampMag(abs(cmdL));
    int magR = clampMag(abs(cmdR));
    digitalWrite(DIR_L, (cmdL >= 0) ? HIGH : LOW);
    digitalWrite(DIR_R, (cmdR >= 0) ? HIGH : LOW);
    analogWrite(PWM_L, magL);
    analogWrite(PWM_R, magR);
}

float readOmegaZ_IMU()
{
    imu::Vector<3> g = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    return g.z();
}

// ================== API FUNCTIONS ==================

void init_chassis()
{
    pinMode(ENC_L_A, INPUT_PULLUP);
    pinMode(ENC_L_B, INPUT_PULLUP);
    pinMode(ENC_R_A, INPUT_PULLUP);
    pinMode(ENC_R_B, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrEncL_A, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrEncR_A, CHANGE);

    pinMode(PWM_L, OUTPUT);
    pinMode(DIR_L, OUTPUT);
    pinMode(PWM_R, OUTPUT);
    pinMode(DIR_R, OUTPUT);
    setMotorPWMDIR(0, 0);

    Wire.begin();
    if (!bno.begin())
    {
        Serial.println("ERROR: BNO055 not detected.");
        while (1)
            delay(10);
    }
    bno.setExtCrystalUse(true);

    Serial.println("Calibrating IMU Gyroscope... DO NOT MOVE ROBOT.");

    uint8_t system, gyro, accel, mag;
    system = gyro = accel = mag = 0;

    while (gyro < 3)
    {
        bno.getCalibration(&system, &gyro, &accel, &mag);
        Serial.print("Calibration Scores - Sys: ");
        Serial.print(system);
        Serial.print(" G: ");
        Serial.print(gyro);
        Serial.print(" A: ");
        Serial.print(accel);
        Serial.print(" M: ");
        Serial.println(mag);
        delay(200);
    }

    Serial.println("IMU Gyro Fully Calibrated!");
    Serial.println(">>> YAW CONVENTION CHECK: Turn robot LEFT (CCW from above).");
    Serial.println(">>> You should see POSITIVE gyro Z values below.");
    Serial.println(">>> If negative, flip the sign in readOmegaZ_IMU().");
    for (int i = 0; i < 25; i++) {  // 5 seconds of readings
        Serial.print("Gyro Z: ");
        Serial.println(readOmegaZ_IMU(), 3);
        delay(200);
    }
    Serial.println(">>> End of yaw check. Safe to start.");

    // Reset odometry to spawn position (world frame, matching simulation)
    odom_x_m = SPAWN_X;
    odom_y_m = SPAWN_Y;
    odom_th_rad = SPAWN_YAW;

    // Conservative starting PID gains
    pidSpeed.Kp = 150.0f;
    pidSpeed.Ki = 40.0f;
    pidSpeed.Kd = 0.0f;
    pidOmega.Kp = 80.0f;
    pidOmega.Ki = 0.0f;
    pidOmega.Kd = 5.0f;

    // Initialize command watchdog
    last_command_time_ms = millis();
}

void update_chassis()
{
    static uint32_t lastMs = 0;
    uint32_t now = millis();
    if (now - lastMs < CTRL_DT_MS)
        return;

    float dt = (float)(now - lastMs) / 1000.0f;
    uint32_t elapsed_ms = now - lastMs; // Save BEFORE updating lastMs
    lastMs = now;

    // 1. Read Encoders
    noInterrupts();
    int32_t tL = ticksL;
    int32_t tR = ticksR;
    interrupts();

    int32_t dL = tL - prevTicksL;
    int32_t dR = tR - prevTicksR;
    prevTicksL = tL;
    prevTicksR = tR;

    float vL = (dL * METERS_PER_COUNT) / dt;
    float vR = (dR * METERS_PER_COUNT) / dt;
    float vAvg = 0.5f * (vL + vR);

    // 2. Sensor Fusion for Turn Rate
    float omegaEnc = (vR - vL) / TRACK_WIDTH_M;
    float omegaImu = readOmegaZ_IMU();
    float omegaFused = ALPHA * omegaImu + (1.0f - ALPHA) * omegaEnc;

    // 3. Update Pose
    odom_th_rad = wrapPi(odom_th_rad + omegaFused * dt);
    odom_x_m += vAvg * cosf(odom_th_rad) * dt;
    odom_y_m += vAvg * sinf(odom_th_rad) * dt;

    // 4. PID Control
    float uV = pidSpeed.update(current_vTarget - vAvg, dt);
    float uW = pidOmega.update(current_omegaTarget - omegaFused, dt);

    int cmdL = (int)lround(uV - uW);
    int cmdR = (int)lround(uV + uW);

    // --- SAFETY: Command watchdog ---
    if (millis() - last_command_time_ms > 500)
    {
        emergency_stop();
        Serial.println("SAFETY: No command for 500ms. Stopping.");
        return;
    }

    // --- SAFETY: Stall detection ---
    float cmdMag = 0.5f * (abs(cmdL) + abs(cmdR));
    if (cmdMag > 40 && fabs(vAvg) < 0.02f)
    {
        stall_timer_ms += elapsed_ms;
        if (stall_timer_ms > STALL_TIMEOUT_MS)
        {
            Serial.println("SAFETY: Motor stall detected! Killing motors.");
            emergency_stop();
            return;
        }
    }
    else
    {
        stall_timer_ms = 0;
    }

    // --- Proportional PWM scaling (preserves turn ratio) ---
    int maxCmd = max(abs(cmdL), abs(cmdR));
    if (maxCmd > PWM_LIMIT)
    {
        float scale = (float)PWM_LIMIT / (float)maxCmd;
        cmdL = (int)lround(cmdL * scale);
        cmdR = (int)lround(cmdR * scale);
    }

    setMotorPWMDIR(cmdL, cmdR);
}

void execute_motor_command(int action)
{
    last_command_time_ms = millis();
    V_TURN = V_FWD / 6.0f;

    if (action == 0)
    { // FWD
        current_vTarget = V_FWD;
        current_omegaTarget = 0.0f;
    }
    else if (action == 1)
    { // LEFT
        current_vTarget = V_TURN;
        current_omegaTarget = OMEGA_TURN;
    }
    else if (action == 2)
    { // RIGHT
        current_vTarget = V_TURN;
        current_omegaTarget = -OMEGA_TURN;
    }
    else if (action == -1)
    { // STOP
        current_vTarget = 0.0f;
        current_omegaTarget = 0.0f;
    }
}

void emergency_stop()
{
    current_vTarget = 0.0f;
    current_omegaTarget = 0.0f;
    pidSpeed.reset();
    pidOmega.reset();
    setMotorPWMDIR(0, 0);
    stall_timer_ms = 0;
}

void get_current_pose(float &x, float &y, float &yaw)
{
    x = odom_x_m;
    y = odom_y_m;
    yaw = odom_th_rad;
}

bool is_target_reached()
{
    float dist = hypot(TARGET_X - odom_x_m, TARGET_Y - odom_y_m);
    return (dist < TARGET_RADIUS);
}

void reset_odometry()
{
    // Reset to spawn position (world frame), not (0,0).
    // This ensures target distance/angle calculations match simulation.
    odom_x_m = SPAWN_X;
    odom_y_m = SPAWN_Y;

    noInterrupts();
    ticksL = 0;
    ticksR = 0;
    interrupts();
    prevTicksL = 0;
    prevTicksR = 0;

    // Reset yaw to spawn heading
    odom_th_rad = SPAWN_YAW;

    pidSpeed.reset();
    pidOmega.reset();
    stall_timer_ms = 0;
    last_command_time_ms = millis();
}