/*
 * FEEDFORWARD PWM CALIBRATION TEST
 * ==================================
 * Maps PWM values to measured wheel velocities so you can set
 * FF_PWM_FWD, FF_PWM_TURN, and FF_PWM_OMEGA in chassis.cpp.
 *
 * Uses the SAME encoder constants, ISRs, pin definitions, velocity
 * filter (alpha=0.3), and PWM_LIMIT (120) as the main firmware.
 * Does NOT require IMU — encoders only.
 *
 * SAFETY: ELEVATE THE ROBOT — wheels must be OFF the ground!
 *
 * THREE MODES (run sequentially):
 *
 *   Mode 1: SYMMETRIC SWEEP
 *     Both motors at same PWM, sweep 20→120, step 10.
 *     Finds FF_PWM_FWD (the PWM that produces your target V_FWD).
 *     Also reveals the motor deadzone.
 *
 *   Mode 2: DIFFERENTIAL SWEEP
 *     Tests rotation at various differentials, with base=0 (pure rotation)
 *     and base=40 (rotation + forward bias).
 *     Finds FF_PWM_OMEGA and answers whether V_TURN > 0 is achievable.
 *
 *   Mode 3: TURN PROFILE VALIDATION
 *     Runs the actual FF values configured below and compares
 *     measured velocity/omega against simulation targets.
 *
 * HOW TO USE:
 *   1. Elevate robot (wheels in the air)
 *   2. Upload to Portenta H7, open Serial Monitor at 115200
 *   3. Read Mode 1 table → set FF_PWM_FWD below
 *   4. Read Mode 2 table → set FF_PWM_OMEGA below, decide FF_PWM_TURN
 *   5. Update the constants below, re-upload, verify Mode 3
 */

#include "mbed.h"
#include <Arduino.h>

// ================== USER CONFIGURATION ==================
// Set these to your best-guess FF values. After running Mode 1 & 2,
// update them and re-upload to validate with Mode 3.
static int FF_PWM_FWD   = 55;   // PWM for ~V_FWD (0.18 m/s)
static int FF_PWM_TURN  = 40;   // Symmetric speed during turns
static int FF_PWM_OMEGA = 40;   // Differential for ~OMEGA_TURN (0.5 rad/s)

// Target velocities (must match simulation action definitions)
static constexpr float TARGET_V_FWD  = 0.18f;  // m/s
static constexpr float TARGET_V_TURN = 0.03f;   // m/s
static constexpr float TARGET_OMEGA  = 0.50f;   // rad/s

// Sweep configuration
static constexpr int SWEEP_MIN  = 20;
static constexpr int SWEEP_MAX  = 120;
static constexpr int SWEEP_STEP = 10;

// Timing (ms)
static constexpr int RUN_DURATION_MS    = 3000;  // Total per step
static constexpr int SETTLE_MS          = 1500;  // Discard initial transient
static constexpr int PAUSE_BETWEEN_MS   = 1000;  // Motor cool-down between steps

// ================== HARDWARE CONFIG (must match chassis.cpp) ==================

const int PWM_L = 6;
const int DIR_L = 5;
const int PWM_R = 2;
const int DIR_R = 1;

// Encoder pins
mbed::InterruptIn encR_A(PD_4);
mbed::DigitalIn   encR_B(PG_10);
mbed::InterruptIn encL_A(PC_7);
mbed::DigitalIn   encL_B(PH_15);

// Constants (identical to chassis.cpp)
static constexpr int   PWM_LIMIT        = 120;
static constexpr float TRACK_WIDTH_M    = 0.26f;
static constexpr float CPR_MOTOR        = 48.0f;
static constexpr float GEAR_RATIO_EXACT = 34.014f;
static constexpr float CPR_WHEEL        = CPR_MOTOR * GEAR_RATIO_EXACT;
static constexpr float WHEEL_DIAM_M     = 0.12f;
static constexpr float WHEEL_CIRC_M     = 3.1415926f * WHEEL_DIAM_M;
static constexpr float METERS_PER_COUNT = WHEEL_CIRC_M / CPR_WHEEL;
static constexpr float VEL_FILTER_ALPHA = 0.3f;
static constexpr uint32_t SAMPLE_DT_MS  = 20;  // 50Hz, same as CTRL_DT_MS

// ================== ENCODER ISRs (same as chassis.cpp) ==================

volatile int32_t ticksL = 0;
volatile int32_t ticksR = 0;

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

// ================== MOTOR CONTROL (same as chassis.cpp) ==================

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

// ================== MEASUREMENT ==================

struct MeasureResult
{
    float vL;     // Filtered left velocity (m/s)
    float vR;     // Filtered right velocity (m/s)
    float vAvg;   // Average linear velocity (m/s)
    float omega;  // Yaw rate from encoders (rad/s)
};

// Runs motors at (cmdL, cmdR) for total_ms. Discards the first settle_ms
// to let transients die out, then averages filtered velocity over the
// remaining measurement window. Uses the same 50Hz + LPF as chassis.cpp.
MeasureResult measureSteadyState(int cmdL, int cmdR,
                                 int total_ms = RUN_DURATION_MS,
                                 int settle_ms = SETTLE_MS)
{
    // Reset filter and encoder state
    float vL_filt = 0.0f;
    float vR_filt = 0.0f;

    noInterrupts();
    int32_t prevL = ticksL;
    int32_t prevR = ticksR;
    interrupts();

    // Start motors
    setMotorPWMDIR(cmdL, cmdR);

    float sumVL = 0.0f, sumVR = 0.0f;
    int sampleCount = 0;
    int livePrintCounter = 0;

    uint32_t startMs = millis();
    uint32_t lastSampleMs = startMs;

    while (millis() - startMs < (uint32_t)total_ms)
    {
        uint32_t now = millis();
        if (now - lastSampleMs < SAMPLE_DT_MS)
            continue;

        float dt = (float)(now - lastSampleMs) / 1000.0f;
        lastSampleMs = now;

        // Read encoders
        noInterrupts();
        int32_t tL = ticksL;
        int32_t tR = ticksR;
        interrupts();

        int32_t dL = tL - prevL;
        int32_t dR = tR - prevR;
        prevL = tL;
        prevR = tR;

        // Velocity calculation with LPF (identical to chassis.cpp)
        float vL_raw = (dL * METERS_PER_COUNT) / dt;
        float vR_raw = (dR * METERS_PER_COUNT) / dt;
        vL_filt = VEL_FILTER_ALPHA * vL_raw + (1.0f - VEL_FILTER_ALPHA) * vL_filt;
        vR_filt = VEL_FILTER_ALPHA * vR_raw + (1.0f - VEL_FILTER_ALPHA) * vR_filt;

        uint32_t elapsed = now - startMs;

        // Accumulate only after settle period
        if (elapsed >= (uint32_t)settle_ms)
        {
            sumVL += vL_filt;
            sumVR += vR_filt;
            sampleCount++;
        }

        // Print live reading every ~200ms for visibility
        livePrintCounter++;
        if (livePrintCounter >= 10) // 10 * 20ms = 200ms
        {
            livePrintCounter = 0;
            const char* phase = (elapsed < (uint32_t)settle_ms) ? "SETTLE" : "MEASURE";
            Serial.print("  [");
            Serial.print(phase);
            Serial.print("] vL=");
            Serial.print(vL_filt, 3);
            Serial.print(" vR=");
            Serial.print(vR_filt, 3);
            float omega_live = (vR_filt - vL_filt) / TRACK_WIDTH_M;
            Serial.print(" omega=");
            Serial.println(omega_live, 3);
        }
    }

    // Stop motors
    setMotorPWMDIR(0, 0);

    // Compute averages
    MeasureResult r;
    if (sampleCount > 0)
    {
        r.vL = sumVL / sampleCount;
        r.vR = sumVR / sampleCount;
    }
    else
    {
        r.vL = 0.0f;
        r.vR = 0.0f;
    }
    r.vAvg = 0.5f * (r.vL + r.vR);
    r.omega = (r.vR - r.vL) / TRACK_WIDTH_M;
    return r;
}

// ================== MODE 1: SYMMETRIC SWEEP ==================

void runMode1()
{
    Serial.println("========================================");
    Serial.println("MODE 1: SYMMETRIC SWEEP (both motors same PWM)");
    Serial.println("Purpose: Find FF_PWM_FWD — the PWM that produces V_FWD");
    Serial.print("Target V_FWD = ");
    Serial.print(TARGET_V_FWD, 3);
    Serial.println(" m/s");
    Serial.println("========================================");
    Serial.println(" PWM | vL(m/s)  | vR(m/s)  | vAvg(m/s) | omega(r/s)");
    Serial.println("-----|----------|----------|-----------|----------");

    int bestPWM = 0;
    float bestErr = 999.0f;

    for (int pwm = SWEEP_MIN; pwm <= SWEEP_MAX; pwm += SWEEP_STEP)
    {
        Serial.print(">>> Running PWM=");
        Serial.println(pwm);

        MeasureResult r = measureSteadyState(pwm, pwm);

        // Print table row
        Serial.print(" ");
        if (pwm < 100) Serial.print(" ");
        if (pwm < 10)  Serial.print(" ");
        Serial.print(pwm);
        Serial.print(" |  ");
        Serial.print(r.vL, 3);
        Serial.print("  |  ");
        Serial.print(r.vR, 3);
        Serial.print("  |   ");
        Serial.print(r.vAvg, 3);
        Serial.print("   |   ");
        Serial.println(r.omega, 3);

        // Track best match
        float err = fabs(r.vAvg - TARGET_V_FWD);
        if (err < bestErr)
        {
            bestErr = err;
            bestPWM = pwm;
        }

        // Note deadzone
        if (fabs(r.vAvg) < 0.005f)
        {
            Serial.print("  ^ DEADZONE: PWM=");
            Serial.print(pwm);
            Serial.println(" produced no motion");
        }

        delay(PAUSE_BETWEEN_MS);
    }

    Serial.println("========================================");
    Serial.print("RESULT: For V_FWD=");
    Serial.print(TARGET_V_FWD, 2);
    Serial.print(" m/s, best match is FF_PWM_FWD = ");
    Serial.print(bestPWM);
    Serial.print(" (vAvg=");
    Serial.print(TARGET_V_FWD - bestErr, 3);
    Serial.println(")");
    Serial.println("========================================");
    Serial.println();
}

// ================== MODE 2: DIFFERENTIAL SWEEP ==================

void runMode2()
{
    Serial.println("========================================");
    Serial.println("MODE 2: DIFFERENTIAL SWEEP (find FF_PWM_OMEGA)");
    Serial.print("Purpose: Find differential PWM for OMEGA_TURN = ");
    Serial.print(TARGET_OMEGA, 2);
    Serial.println(" rad/s");
    Serial.println("Also tests whether V_TURN > 0 is achievable with a forward bias.");
    Serial.println("========================================");
    Serial.println("Base | Diff | cmdL | cmdR | vAvg(m/s) | omega(r/s)");
    Serial.println("-----|------|------|------|-----------|----------");

    // --- Part A: Pure rotation (base=0) ---
    Serial.println("--- Pure rotation (base=0) ---");
    int bestDiff_pure = 0;
    float bestOmegaErr_pure = 999.0f;

    for (int diff = 10; diff <= 60; diff += 10)
    {
        int cL = -diff;
        int cR = diff;
        Serial.print(">>> Running base=0 diff=");
        Serial.println(diff);

        MeasureResult r = measureSteadyState(cL, cR);

        Serial.print("   0 |   ");
        if (diff < 10) Serial.print(" ");
        Serial.print(diff);
        Serial.print(" |  ");
        if (cL >= 0) Serial.print(" ");
        Serial.print(cL);
        Serial.print(" |   ");
        if (cR < 100) Serial.print(" ");
        Serial.print(cR);
        Serial.print(" |   ");
        Serial.print(r.vAvg, 3);
        Serial.print("   |   ");
        Serial.println(r.omega, 3);

        float omegaErr = fabs(fabs(r.omega) - TARGET_OMEGA);
        if (omegaErr < bestOmegaErr_pure)
        {
            bestOmegaErr_pure = omegaErr;
            bestDiff_pure = diff;
        }

        delay(PAUSE_BETWEEN_MS);
    }

    // --- Part B: With forward bias (base=40) ---
    Serial.println("--- With forward bias (base=40) ---");
    for (int diff = 10; diff <= 50; diff += 10)
    {
        int base = 40;
        int cL = base - diff;
        int cR = base + diff;
        Serial.print(">>> Running base=40 diff=");
        Serial.println(diff);

        MeasureResult r = measureSteadyState(cL, cR);

        Serial.print("  40 |   ");
        if (diff < 10) Serial.print(" ");
        Serial.print(diff);
        Serial.print(" |  ");
        if (cL >= 0) Serial.print(" ");
        if (abs(cL) < 10) Serial.print(" ");
        Serial.print(cL);
        Serial.print(" |   ");
        if (cR < 100) Serial.print(" ");
        Serial.print(cR);
        Serial.print(" |   ");
        Serial.print(r.vAvg, 3);
        Serial.print("   |   ");
        Serial.println(r.omega, 3);

        delay(PAUSE_BETWEEN_MS);
    }

    Serial.println("========================================");
    Serial.print("RESULT: For pure rotation at OMEGA=");
    Serial.print(TARGET_OMEGA, 2);
    Serial.print(" rad/s, best match is FF_PWM_OMEGA = ");
    Serial.println(bestDiff_pure);
    Serial.println("Check base=40 rows: if vAvg > 0.02, V_TURN > 0 is achievable.");
    Serial.println("If vAvg ~ 0 for all base=40 rows, set V_TURN=0 and FF_PWM_TURN=0.");
    Serial.println("========================================");
    Serial.println();
}

// ================== MODE 3: TURN PROFILE VALIDATION ==================

void runMode3()
{
    Serial.println("========================================");
    Serial.println("MODE 3: TURN PROFILE VALIDATION");
    Serial.print("FF values: FWD=");
    Serial.print(FF_PWM_FWD);
    Serial.print(" TURN=");
    Serial.print(FF_PWM_TURN);
    Serial.print(" OMEGA=");
    Serial.println(FF_PWM_OMEGA);
    Serial.println("========================================");
    Serial.println("Action   | cmdL | cmdR | vAvg(m/s) | omega(r/s) | tgt_v | tgt_w | err_v | err_w");
    Serial.println("---------|------|------|-----------|------------|-------|-------|-------|------");

    struct ActionDef
    {
        const char* name;
        int cmdL, cmdR;
        float tgt_v, tgt_w;
    };

    ActionDef actions[3] = {
        {"FORWARD ",
         FF_PWM_FWD, FF_PWM_FWD,
         TARGET_V_FWD, 0.0f},
        {"LEFT    ",
         FF_PWM_TURN - FF_PWM_OMEGA, FF_PWM_TURN + FF_PWM_OMEGA,
         TARGET_V_TURN, TARGET_OMEGA},
        {"RIGHT   ",
         FF_PWM_TURN + FF_PWM_OMEGA, FF_PWM_TURN - FF_PWM_OMEGA,
         TARGET_V_TURN, -TARGET_OMEGA}
    };

    for (int i = 0; i < 3; i++)
    {
        ActionDef& a = actions[i];
        Serial.print(">>> Running ");
        Serial.println(a.name);

        MeasureResult r = measureSteadyState(a.cmdL, a.cmdR);

        float err_v = r.vAvg - a.tgt_v;
        float err_w = r.omega - a.tgt_w;

        Serial.print(a.name);
        Serial.print(" |  ");
        if (a.cmdL >= 0) Serial.print(" ");
        if (abs(a.cmdL) < 10) Serial.print(" ");
        Serial.print(a.cmdL);
        Serial.print(" |  ");
        if (a.cmdR >= 0) Serial.print(" ");
        if (abs(a.cmdR) < 10) Serial.print(" ");
        Serial.print(a.cmdR);
        Serial.print(" |   ");
        Serial.print(r.vAvg, 3);
        Serial.print("   |    ");
        Serial.print(r.omega, 3);
        Serial.print("   | ");
        Serial.print(a.tgt_v, 2);
        Serial.print("  | ");
        Serial.print(a.tgt_w, 2);
        Serial.print("  | ");
        if (err_v >= 0) Serial.print("+");
        Serial.print(err_v, 3);
        Serial.print(" | ");
        if (err_w >= 0) Serial.print("+");
        Serial.println(err_w, 3);

        // Pass/fail
        bool pass_v = fabs(err_v) < 0.05f;
        bool pass_w = fabs(err_w) < 0.15f;
        Serial.print("  -> v: ");
        Serial.print(pass_v ? "PASS" : "FAIL");
        Serial.print("  w: ");
        Serial.println(pass_w ? "PASS" : "FAIL");

        delay(PAUSE_BETWEEN_MS);
    }

    Serial.println("========================================");
    Serial.println("If FAIL: adjust FF values at top of file and re-upload.");
    Serial.println("========================================");
}

// ================== SETUP & LOOP ==================

void setup()
{
    Serial.begin(115200);
    while (!Serial && millis() < 3000)
        ;

    // Motor pins
    pinMode(PWM_L, OUTPUT);
    pinMode(DIR_L, OUTPUT);
    pinMode(PWM_R, OUTPUT);
    pinMode(DIR_R, OUTPUT);
    setMotorPWMDIR(0, 0);

    // Encoder pins
    encL_A.mode(PullUp);
    encL_B.mode(PullUp);
    encR_A.mode(PullUp);
    encR_B.mode(PullUp);

    encL_A.rise(&isrEncL_A);
    encL_A.fall(&isrEncL_A);
    encR_A.rise(&isrEncR_A);
    encR_A.fall(&isrEncR_A);

    Serial.println("========================================");
    Serial.println("FEEDFORWARD PWM CALIBRATION TEST");
    Serial.println("!! ELEVATE ROBOT — wheels must be OFF the ground !!");
    Serial.print("PWM_LIMIT = ");
    Serial.print(PWM_LIMIT);
    Serial.print("  VEL_FILTER_ALPHA = ");
    Serial.println(VEL_FILTER_ALPHA, 1);
    Serial.println("Starting in 5 seconds...");
    Serial.println("========================================");
    delay(5000);

    // Run all three modes
    runMode1();
    delay(3000);
    runMode2();
    delay(3000);
    runMode3();

    // Final summary
    Serial.println();
    Serial.println("========================================");
    Serial.println("CALIBRATION COMPLETE");
    Serial.println();
    Serial.println("Next steps:");
    Serial.println("  1. Note the best FF_PWM_FWD from Mode 1");
    Serial.println("  2. Note the best FF_PWM_OMEGA from Mode 2");
    Serial.println("  3. Check Mode 2 base=40 rows to decide V_TURN");
    Serial.println("  4. Update FF values at top of this file");
    Serial.println("  5. Re-upload and check Mode 3 passes");
    Serial.println("  6. Copy final values to chassis.cpp");
    Serial.println("========================================");
}

void loop()
{
    // Test runs once in setup()
}
