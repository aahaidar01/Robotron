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

// ================== FEEDFORWARD + PI TRIM ARCHITECTURE ==================
// Feedforward provides the bulk of the motor command. PI only trims around it.
// This reduces deadzone issues, integral windup, and current spikes vs. pure PID.

// RL Agent Target Speeds (what the sim expects)
float current_vTarget = 0.0f;
float current_omegaTarget = 0.0f;

// Base Speeds (must match simulation action definitions)
float V_FWD = 0.18f;
float V_TURN = 0.03f;
static constexpr float OMEGA_TURN = 0.5f;

// --- Feedforward PWM values (TUNE THESE ON HARDWARE) ---
static int FF_PWM_FWD   = 55;  // PWM that produces ~V_FWD (0.18 m/s)
static int FF_PWM_TURN  = 40;  // PWM that produces ~V_TURN (0.03 m/s) during turns
static int FF_PWM_OMEGA = 40;  // Differential PWM that produces ~OMEGA_TURN (0.5 rad/s)

// Current feedforward base values (set by execute_motor_command)
static int ff_base_speed = 0;  // Symmetric component (both motors)
static int ff_base_omega = 0;  // Differential component (left-right split)

// --- Global Odometry Variables ---
static float odom_x_m = 0.0f;
static float odom_y_m = 0.0f;
static float odom_th_rad = 0.0f;

// --- Safety Variables ---
static uint32_t stall_timer_ms = 0;
const uint32_t STALL_TIMEOUT_MS = 800;
static uint32_t last_command_time_ms = 0;
bool is_crashed = false; // Tracks terminal state

// --- Slew Rate Limiter State (file-scope so emergency_stop can reset) ---
static int prev_cmdL = 0;
static int prev_cmdR = 0;

// --- Ramp-aware anti-windup state ---
static bool hold_speed_integrator = false;
static bool hold_omega_integrator = false;

// --- Velocity Low-Pass Filter State ---
static float vL_filt = 0.0f;
static float vR_filt = 0.0f;
static constexpr float VEL_FILTER_ALPHA = 0.3f; // 0.0 = full smoothing, 1.0 = no filter

// --- Debug Snapshot (updated every PID cycle, read by log_chassis_state) ---
static float dbg_vL_raw = 0, dbg_vR_raw = 0;
static float dbg_vL = 0, dbg_vR = 0, dbg_vAvg = 0;
static float dbg_omegaImu = 0, dbg_omegaEnc = 0, dbg_omegaFused = 0;
static float dbg_trimV = 0, dbg_trimW = 0;
static int   dbg_cmdL = 0, dbg_cmdR = 0;
static int   dbg_ffL = 0, dbg_ffR = 0; // Feedforward component for debugging

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

    void setOutputLimits(float minOut, float maxOut)
    {
        outMin = minOut;
        outMax = maxOut;
    }

    float update(float err, float dt, bool allowIntegrate = true)
    {
        if (dt <= 1e-6f)
            return 0.0f;

        // Zero-crossing decay — retain some integral memory while damping overshoot.
        if ((err > 0 && prevErr < 0) || (err < 0 && prevErr > 0))
        {
            integ *= 0.3f;
        }

        float deriv = (err - prevErr) / dt;

        if (allowIntegrate)
        {
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
        }

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

static inline int applySlewLimit(int target, int prev)
{
    // Softer acceleration from rest / reversals, faster decel for safety.
    static constexpr int ACCEL_STEP = 15;
    static constexpr int DECEL_STEP = 30;

    int step = DECEL_STEP;

    if (target == prev)
        return target;

    if ((prev == 0 && target != 0) || (prev > 0 && target < 0) || (prev < 0 && target > 0))
    {
        step = ACCEL_STEP;
    }
    else if (abs(target) > abs(prev))
    {
        step = ACCEL_STEP;
    }
    else
    {
        step = DECEL_STEP;
    }

    if (target > prev + step)
        return prev + step;
    if (target < prev - step)
        return prev - step;
    return target;
}

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
    for (int i = 0; i < 25; i++)
    {
        Serial.print("Gyro Z: ");
        Serial.println(readOmegaZ_IMU(), 3);
        delay(200);
    }
    Serial.println(">>> End of yaw check. Safe to start.");

    // Reset odometry to spawn position (world frame, matching simulation)
    odom_x_m = SPAWN_X;
    odom_y_m = SPAWN_Y;
    odom_th_rad = SPAWN_YAW;

    // --- PI Trim Gains (low — only correcting small ff errors) ---
    pidSpeed.Kp = 60.0f;
    pidSpeed.Ki = 15.0f;
    pidSpeed.Kd = 0.0f;
    pidSpeed.setOutputLimits(-30.0f, 30.0f);

    pidOmega.Kp = 40.0f;
    pidOmega.Ki = 10.0f;
    pidOmega.Kd = 0.0f;
    pidOmega.setOutputLimits(-25.0f, 25.0f);

    hold_speed_integrator = false;
    hold_omega_integrator = false;

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
    if (lastMs == 0)
    {
        lastMs = now;
        prevTicksL = ticksL;
        prevTicksR = ticksR;
        return;
    }

    if (now - lastMs < CTRL_DT_MS)
        return;

    float dt = (float)(now - lastMs) / 1000.0f;
    if (dt > 0.1f)
        dt = 0.1f; // Cap at 100ms
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

    float vL_raw = (dL * METERS_PER_COUNT) / dt;
    float vR_raw = (dR * METERS_PER_COUNT) / dt;
    vL_filt = VEL_FILTER_ALPHA * vL_raw + (1.0f - VEL_FILTER_ALPHA) * vL_filt;
    vR_filt = VEL_FILTER_ALPHA * vR_raw + (1.0f - VEL_FILTER_ALPHA) * vR_filt;
    float vL = vL_filt;
    float vR = vR_filt;
    float vAvg = 0.5f * (vL + vR);

    // 2. Sensor Fusion for Turn Rate
    float omegaEnc = (vR - vL) / TRACK_WIDTH_M;
    float omegaImu = readOmegaZ_IMU();
    float omegaFused = ALPHA * omegaImu + (1.0f - ALPHA) * omegaEnc;

    // 3. Update Pose
    odom_th_rad = wrapPi(odom_th_rad + omegaFused * dt);
    odom_x_m += vAvg * cosf(odom_th_rad) * dt;
    odom_y_m += vAvg * sinf(odom_th_rad) * dt;

    // 4. Feedforward + PI Trim Control
    // If the previous cycle hit actuator limits or the slew limiter,
    // hold the integrators this cycle. This is simple ramp-aware anti-windup.
    float trimV = pidSpeed.update(current_vTarget - vAvg, dt, !hold_speed_integrator);
    float trimW = pidOmega.update(current_omegaTarget - omegaFused, dt, !hold_omega_integrator);

    int ffL = ff_base_speed - ff_base_omega;
    int ffR = ff_base_speed + ff_base_omega;

    int cmdL_raw = ffL + (int)lround(trimV - trimW);
    int cmdR_raw = ffR + (int)lround(trimV + trimW);

    // --- SAFETY: Command watchdog ---
    if (millis() - last_command_time_ms > 2000)
    {
        emergency_stop();
        Serial.println("SAFETY: No command for 2000ms. Stopping.");
        return;
    }

    // --- SAFETY: Stall detection (per-motor) ---
    // Catches single-side stalls that vAvg averaging would miss.
    bool stalled = false;
    if (abs(cmdL_raw) > 50 && fabs(vL) < 0.02f) stalled = true;
    if (abs(cmdR_raw) > 50 && fabs(vR) < 0.02f) stalled = true;

    if (stalled)
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
    int cmdL_scaled = cmdL_raw;
    int cmdR_scaled = cmdR_raw;
    bool mixedLimited = false;

    int maxCmd = max(abs(cmdL_scaled), abs(cmdR_scaled));
    if (maxCmd > PWM_LIMIT)
    {
        float scale = (float)PWM_LIMIT / (float)maxCmd;
        cmdL_scaled = (int)lround(cmdL_scaled * scale);
        cmdR_scaled = (int)lround(cmdR_scaled * scale);
        mixedLimited = true;
    }

    // --- Slew Rate Limiter ---
    // Softer acceleration than deceleration to reduce inrush current and gearbox shock.
    int cmdL = applySlewLimit(cmdL_scaled, prev_cmdL);
    int cmdR = applySlewLimit(cmdR_scaled, prev_cmdR);
    bool slewLimited = (cmdL != cmdL_scaled) || (cmdR != cmdR_scaled);

    prev_cmdL = cmdL;
    prev_cmdR = cmdR;

    // --- Ramp-aware anti-windup flags for next cycle ---
    hold_speed_integrator = mixedLimited || slewLimited;
    hold_omega_integrator = mixedLimited || slewLimited;

    // --- Store debug snapshot for log_chassis_state() ---
    dbg_vL_raw = vL_raw;
    dbg_vR_raw = vR_raw;
    dbg_vL = vL;
    dbg_vR = vR;
    dbg_vAvg = vAvg;
    dbg_omegaImu = omegaImu;
    dbg_omegaEnc = omegaEnc;
    dbg_omegaFused = omegaFused;
    dbg_trimV = trimV;
    dbg_trimW = trimW;
    dbg_cmdL = cmdL;
    dbg_cmdR = cmdR;
    dbg_ffL = ffL;
    dbg_ffR = ffR;

    setMotorPWMDIR(cmdL, cmdR);
}

void execute_motor_command(int action)
{
    last_command_time_ms = millis();

    if (is_crashed)
    {
        return;
    }

    // Reset PI trim on action change — prevents stale integral
    // and derivative kick from causing jitter at transitions.
    // First PID cycle after change produces trim=0 (pure feedforward).
    static int prev_action = -1;
    if (action != prev_action)
    {
        pidSpeed.reset();
        pidOmega.reset();
        hold_speed_integrator = false;
        hold_omega_integrator = false;
        prev_action = action;
    }

    if (action == 0)
    { // FWD
        current_vTarget = V_FWD;
        current_omegaTarget = 0.0f;
        ff_base_speed = FF_PWM_FWD;
        ff_base_omega = 0;
    }
    else if (action == 1)
    { // LEFT (positive omega = CCW)
        current_vTarget = V_TURN;
        current_omegaTarget = OMEGA_TURN;
        ff_base_speed = FF_PWM_TURN;
        ff_base_omega = FF_PWM_OMEGA;
    }
    else if (action == 2)
    { // RIGHT (negative omega = CW)
        current_vTarget = V_TURN;
        current_omegaTarget = -OMEGA_TURN;
        ff_base_speed = FF_PWM_TURN;
        ff_base_omega = -FF_PWM_OMEGA;
    }
    else if (action == -1)
    { // STOP
        current_vTarget = 0.0f;
        current_omegaTarget = 0.0f;
        ff_base_speed = 0;
        ff_base_omega = 0;
    }
}

void emergency_stop()
{
    current_vTarget = 0.0f;
    current_omegaTarget = 0.0f;
    ff_base_speed = 0;
    ff_base_omega = 0;
    pidSpeed.reset();
    pidOmega.reset();
    hold_speed_integrator = false;
    hold_omega_integrator = false;
    prev_cmdL = 0;
    prev_cmdR = 0;
    vL_filt = 0.0f;
    vR_filt = 0.0f;
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

bool is_chassis_stalled()
{
    return is_crashed;
}

void reset_odometry()
{
    // Stop motors immediately so reset never leaves the previous PWM applied.
    setMotorPWMDIR(0, 0);

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
    hold_speed_integrator = false;
    hold_omega_integrator = false;
    ff_base_speed = 0;
    ff_base_omega = 0;
    prev_cmdL = 0;
    prev_cmdR = 0;
    vL_filt = 0.0f;
    vR_filt = 0.0f;
    stall_timer_ms = 0;
    last_command_time_ms = millis();
    is_crashed = false;
}

void log_chassis_state(int level)
{
    if (level <= 0)
        return;

    if (level == 1)
    {
        Serial.print("xy(");
        Serial.print(odom_x_m, 2);
        Serial.print(",");
        Serial.print(odom_y_m, 2);
        Serial.print(") yaw:");
        Serial.print(odom_th_rad, 2);
        Serial.print(" v:");
        Serial.print(dbg_vAvg, 2);
        Serial.print(" w:");
        Serial.print(dbg_omegaFused, 2);
        Serial.print(" crash:");
        Serial.print(is_crashed ? 1 : 0);
    }
    else
    {
        Serial.print("[CHS] pose: (");
        Serial.print(odom_x_m, 3);
        Serial.print(", ");
        Serial.print(odom_y_m, 3);
        Serial.print(") yaw: ");
        Serial.print(odom_th_rad, 3);
        Serial.print(" rad (");
        Serial.print(odom_th_rad * 180.0f / 3.1415926f, 1);
        Serial.println(" deg)");

        Serial.print("[CHS] wheels: vL=");
        Serial.print(dbg_vL, 3);
        Serial.print("(raw:");
        Serial.print(dbg_vL_raw, 3);
        Serial.print(") vR=");
        Serial.print(dbg_vR, 3);
        Serial.print("(raw:");
        Serial.print(dbg_vR_raw, 3);
        Serial.print(") vAvg=");
        Serial.print(dbg_vAvg, 3);
        Serial.print(" | tgt_v=");
        Serial.print(current_vTarget, 3);
        Serial.print(" tgt_w=");
        Serial.println(current_omegaTarget, 3);

        Serial.print("[CHS] omega: imu=");
        Serial.print(dbg_omegaImu, 3);
        Serial.print(" enc=");
        Serial.print(dbg_omegaEnc, 3);
        Serial.print(" fused=");
        Serial.println(dbg_omegaFused, 3);

        Serial.print("[CHS] ff: L=");
        Serial.print(dbg_ffL);
        Serial.print(" R=");
        Serial.print(dbg_ffR);
        Serial.print(" | trim: V=");
        Serial.print(dbg_trimV, 1);
        Serial.print(" W=");
        Serial.println(dbg_trimW, 1);

        Serial.print("[CHS] final cmd: L=");
        Serial.print(dbg_cmdL);
        Serial.print(" R=");
        Serial.print(dbg_cmdR);
        Serial.print(" crash:");
        Serial.print(is_crashed ? 1 : 0);
        Serial.print(" | stall: ");
        Serial.println(stall_timer_ms);

        noInterrupts();
        int32_t tL = ticksL;
        int32_t tR = ticksR;
        interrupts();
        Serial.print("[CHS] ticks: L=");
        Serial.print(tL);
        Serial.print(" R=");
        Serial.println(tR);
    }
}
