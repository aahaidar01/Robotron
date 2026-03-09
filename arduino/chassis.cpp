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

// ---- Mbed Pin Objects ----
// Right Encoder
mbed::InterruptIn encR_A(PD_4);
mbed::DigitalIn   encR_B(PG_10);

// Left Encoder (Using standard Arduino pins 4 and 0 via Mbed)
mbed::InterruptIn encL_A(PC_7);  // Arduino Pin 4 resolves to PC_7 on Portenta
mbed::DigitalIn   encL_B(PH_15); // Arduino Pin 0 resolves to PH_15 on Portenta

const int PWM_L = 6;
const int DIR_L = 5;
const int PWM_R = 2;
const int DIR_R = 1;

static constexpr int PWM_LIMIT = 120;
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
static constexpr float ALPHA = 0.5f; // IMU and Encoders evenly trusted

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
bool is_crashed = false; //Tracks terminal state

// --- Debug Snapshot (updated every PID cycle, read by log_chassis_state) ---
static float dbg_vL = 0, dbg_vR = 0, dbg_vAvg = 0;
static float dbg_omegaImu = 0, dbg_omegaEnc = 0, dbg_omegaFused = 0;
static float dbg_uV = 0, dbg_uW = 0;
static int   dbg_cmdL = 0, dbg_cmdR = 0;

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
        // // --- Error Tolerance (Deadzone) ---
        // // If the error is tiny, ignore it to prevent micro-jiggling
        // if (fabs(err) < 0.03f) 
        // {
        //     err = 0.0f;
        // }

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
    // Mbed .read() returns 1 (HIGH) or 0 (LOW)
    bool a = encL_A.read();
    bool b = encL_B.read();
    ticksL += (a == b) ? +1 : -1;
}

void isrEncR_A()
{
    bool a = encR_A.read();
    bool b = encR_B.read();
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
    return g.z() * (3.1415926f / 180.0f);
}

// ================== API FUNCTIONS ==================

void init_chassis()
{
    encL_A.mode(PullUp);
    encL_B.mode(PullUp);
    encR_A.mode(PullUp);
    encR_B.mode(PullUp);

    // Attach interrupts for both RISE and FALL to mimic Arduino's 'CHANGE'
    encL_A.rise(&isrEncL_A);
    encL_A.fall(&isrEncL_A);
    
    encR_A.rise(&isrEncR_A);
    encR_A.fall(&isrEncR_A);

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

    // Heavy Tank PID gains
    pidSpeed.Kp = 250.0f;  
    pidSpeed.Ki = 50.0f;
    pidSpeed.Kd = 0.0f;
    
    pidOmega.Kp = 100.0f;  
    pidOmega.Ki = 20.0f;
    pidOmega.Kd = 0.0f;
    // Initialize command watchdog
    last_command_time_ms = millis();
}

void update_chassis()
{
    static uint32_t lastMs = 0;
    uint32_t now = millis();

    // First call after boot: just set the baseline, don't run PID.
    // Without this, dt = ~15s (all calibration time) which causes
    // motor jitter, instant stall detection, and odometry corruption.
    if (lastMs == 0) {
        lastMs = now;
        prevTicksL = ticksL;
        prevTicksR = ticksR;
        return;
    }

    if (now - lastMs < CTRL_DT_MS)
        return;

    float dt = (float)(now - lastMs) / 1000.0f;
    if (dt > 0.1f) dt = 0.1f;          // Cap at 100ms
    uint32_t elapsed_ms = now - lastMs; 
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

    // --- Store debug snapshot for log_chassis_state() ---
    dbg_vL = vL; dbg_vR = vR; dbg_vAvg = vAvg;
    dbg_omegaImu = omegaImu; dbg_omegaEnc = omegaEnc; dbg_omegaFused = omegaFused;
    dbg_uV = uV; dbg_uW = uW;
    dbg_cmdL = cmdL; dbg_cmdR = cmdR;

    // --- SAFETY: Command watchdog ---
    if (millis() - last_command_time_ms > 2000)
    {
        emergency_stop();
        Serial.println("SAFETY: No command for 2000ms. Stopping.");
        return;
    }

    // --- SAFETY: Stall detection ---
    float cmdMag = 0.5f * (abs(cmdL) + abs(cmdR));
    if (cmdMag > 65 && fabs(vAvg) < 0.02f)
    {
        stall_timer_ms += elapsed_ms;
        if (stall_timer_ms > STALL_TIMEOUT_MS)
        {
            Serial.println("SAFETY: Motor stall detected! Killing motors.");
            is_crashed = true;
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

    // --- Slew Rate Limiter (Acceleration Ramping) ---
    static float prev_cmdL = 0.0f;
    static float prev_cmdR = 0.0f;
    const float MAX_STEP = 20.0f; // Max PWM change per 20ms tick

    if (cmdL > prev_cmdL + MAX_STEP) cmdL = prev_cmdL + MAX_STEP;
    else if (cmdL < prev_cmdL - MAX_STEP) cmdL = prev_cmdL - MAX_STEP;

    if (cmdR > prev_cmdR + MAX_STEP) cmdR = prev_cmdR + MAX_STEP;
    else if (cmdR < prev_cmdR - MAX_STEP) cmdR = prev_cmdR - MAX_STEP;

    prev_cmdL = cmdL;
    prev_cmdR = cmdR;

    setMotorPWMDIR(cmdL, cmdR);
}

void execute_motor_command(int action)
{
    last_command_time_ms = millis();

    if (is_crashed) {
        return; 
    }

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

bool is_chassis_stalled() {
    return is_crashed;
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
    is_crashed = false;
}

void log_chassis_state(int level)
{
    if (level <= 0) return;

    if (level == 1)
    {
        Serial.print("xy("); Serial.print(odom_x_m, 2);
        Serial.print(",");   Serial.print(odom_y_m, 2);
        Serial.print(") yaw:"); Serial.print(odom_th_rad, 2);
        Serial.print(" v:");  Serial.print(dbg_vAvg, 2);
        Serial.print(" w:");  Serial.print(dbg_omegaFused, 2);
        Serial.print(" crash:"); Serial.print(is_crashed ? 1 : 0);
    }
    else
    {
        Serial.print("[CHS] pose: ("); Serial.print(odom_x_m, 3);
        Serial.print(", "); Serial.print(odom_y_m, 3);
        Serial.print(") yaw: "); Serial.print(odom_th_rad, 3);
        Serial.print(" rad ("); Serial.print(odom_th_rad * 180.0f / 3.1415926f, 1);
        Serial.println(" deg)");

        Serial.print("[CHS] wheels: vL="); Serial.print(dbg_vL, 3);
        Serial.print(" vR="); Serial.print(dbg_vR, 3);
        Serial.print(" vAvg="); Serial.print(dbg_vAvg, 3);
        Serial.print(" | tgt_v="); Serial.print(current_vTarget, 3);
        Serial.print(" tgt_w="); Serial.println(current_omegaTarget, 3);

        Serial.print("[CHS] omega: imu="); Serial.print(dbg_omegaImu, 3);
        Serial.print(" enc="); Serial.print(dbg_omegaEnc, 3);
        Serial.print(" fused="); Serial.println(dbg_omegaFused, 3);

        Serial.print("[CHS] pid: uV="); Serial.print(dbg_uV, 1);
        Serial.print(" uW="); Serial.print(dbg_uW, 1);
        Serial.print(" | cmd: L="); Serial.print(dbg_cmdL);
        Serial.print(" R="); Serial.print(dbg_cmdR);
        Serial.print(" crash:"); Serial.print(is_crashed ? 1 : 0);
        Serial.print(" | stall: "); Serial.println(stall_timer_ms);


        noInterrupts();
        int32_t tL = ticksL;
        int32_t tR = ticksR;
        interrupts();
        Serial.print("[CHS] ticks: L="); Serial.print(tL);
        Serial.print(" R="); Serial.println(tR);
    }
}