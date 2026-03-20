/*
 * ETHERNET FEEDFORWARD PWM CALIBRATION TEST
 * ==================================
 * Maps PWM values to measured wheel velocities so you can set
 * FF_PWM_FWD, FF_PWM_TURN, and FF_PWM_OMEGA in chassis.cpp.
 *
 * Uses the SAME encoder constants, ISRs, pin definitions, velocity
 * filter (alpha=0.3), and PWM_LIMIT (120) as the main firmware.
 *
 * Hardware: Portenta H7 with Ethernet (via Vision Shield or Breakout Board).
 *           Connect an Ethernet cable directly between the Portenta and your PC.
 *
 * SAFETY: ELEVATE THE ROBOT — wheels must be OFF the ground!
 *
 * HOW TO USE:
 *   1. Elevate robot (wheels in the air).
 *   2. Connect Ethernet cable between Portenta and PC.
 *   3. Configure your PC's Ethernet adapter with a static IP on the
 *      same subnet (e.g. 192.168.1.100, netmask 255.255.255.0).
 *   4. Upload to Portenta H7.
 *   5. From your PC, connect via TCP: nc 192.168.1.177 23
 *      (or use PuTTY / any TCP terminal on port 23).
 *   6. Send ANY character from the terminal to begin the test.
 *   7. Read Mode 1 table → set FF_PWM_FWD below.
 *   8. Read Mode 2 table → set FF_PWM_OMEGA below, decide FF_PWM_TURN.
 *   9. Update constants below, re-upload, re-trigger to validate Mode 3.
 *  10. Copy final values to chassis.cpp.
 *
 * THREE MODES (run sequentially after trigger):
 *
 *   Mode 1: SYMMETRIC SWEEP
 *     Both motors at same PWM, sweep 20→120, step 10.
 *     Finds FF_PWM_FWD and reveals the motor deadzone.
 *
 *   Mode 2: DIFFERENTIAL SWEEP
 *     Tests pure rotation and rotation + forward bias.
 *     Finds FF_PWM_OMEGA and answers whether V_TURN > 0 is achievable.
 *
 *   Mode 3: TURN PROFILE VALIDATION
 *     Runs the configured FF values and compares measured
 *     velocity/omega against simulation targets.
 */

#include "mbed.h"
#include <Arduino.h>
#include <PortentaEthernet.h>
#include <Ethernet.h>

// ================== ETHERNET CONFIG ==================
// Static IP for the Portenta. Make sure your PC's Ethernet adapter
// is on the same subnet (e.g. 192.168.1.100).
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
IPAddress ip(192, 168, 1, 177);
const int ETH_PORT = 23;

EthernetServer server(ETH_PORT);
EthernetClient client;
static bool welcomeSent = false;

// ================== USER CONFIGURATION ==================
// Set these to your best-guess FF values. After running Mode 1 & 2,
// update them and re-upload to validate with Mode 3.
static int FF_PWM_FWD   = 55;  // PWM for ~V_FWD (0.18 m/s)
static int FF_PWM_TURN  = 0;  // Symmetric speed during turns
static int FF_PWM_OMEGA = 55;  // Differential for ~OMEGA_TURN (0.5 rad/s)

// Target velocities (must match simulation action definitions)
static constexpr float TARGET_V_FWD  = 0.18f;  // m/s
static constexpr float TARGET_V_TURN = 0.03f;  // m/s
static constexpr float TARGET_OMEGA  = 0.50f;  // rad/s

// Sweep configuration
static constexpr int SWEEP_MIN  = 20;
static constexpr int SWEEP_MAX  = 70;
static constexpr int SWEEP_STEP = 10;

// Timing (ms)
static constexpr int RUN_DURATION_MS  = 3000;  // Total per step
static constexpr int SETTLE_MS        = 1500;  // Discard initial transient
static constexpr int PAUSE_BETWEEN_MS = 1000;  // Motor cool-down between steps

// ---- Encoder Pin Definitions ----
mbed::InterruptIn encR_A(PD_4);
mbed::DigitalIn   encR_B(PG_10);
mbed::InterruptIn encL_A(PC_7);
mbed::DigitalIn   encL_B(PH_15);

// ---- Motor Pin Definitions (Pure Mbed) ----
mbed::PwmOut      pwm_L(PG_7);  
mbed::DigitalOut  dir_L(PJ_10);  

// Right Motor (Original D2 and D1)
mbed::PwmOut      pwm_R(PJ_11);
mbed::DigitalOut  dir_R(PK_1);  

// Constants (identical to chassis.cpp)
static constexpr int   PWM_LIMIT        = 60;
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
    // Clamp magnitude to your software PWM_LIMIT (e.g., 120)
    int magL = clampMag(abs(cmdL));
    int magR = clampMag(abs(cmdR));

    // Set directions (1 for forward, 0 for reverse)
    dir_L = (cmdL >= 0) ? 1 : 0;
    dir_R = (cmdR >= 0) ? 1 : 0;

    // FIX: Divide by 255.0f to match standard Arduino analogWrite resolution!
    float dutyL = (float)magL / 255.0f;
    float dutyR = (float)magR / 255.0f;

    // Write duty cycle to hardware timers (0.0 to 1.0)
    pwm_L.write(dutyL);
    pwm_R.write(dutyR);
}

// ================== MEASUREMENT ==================
struct MeasureResult
{
    float vL;    // Filtered left velocity (m/s)
    float vR;    // Filtered right velocity (m/s)
    float vAvg;  // Average linear velocity (m/s)
    float omega; // Yaw rate from encoders (rad/s)
};

// Runs motors at (cmdL, cmdR) for total_ms. Discards the first settle_ms
// to let transients die out, then averages filtered velocity over the
// remaining measurement window. Uses the same 50Hz + LPF as chassis.cpp.
MeasureResult measureSteadyState(Print& out, int cmdL, int cmdR,
                                 int total_ms = RUN_DURATION_MS,
                                 int settle_ms = SETTLE_MS)
{
    float vL_filt = 0.0f;
    float vR_filt = 0.0f;

    noInterrupts();
    int32_t prevL = ticksL;
    int32_t prevR = ticksR;
    interrupts();

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

        noInterrupts();
        int32_t tL = ticksL;
        int32_t tR = ticksR;
        interrupts();

        int32_t dL = tL - prevL;
        int32_t dR = tR - prevR;
        prevL = tL;
        prevR = tR;

        float vL_raw = (dL * METERS_PER_COUNT) / dt;
        float vR_raw = (dR * METERS_PER_COUNT) / dt;
        vL_filt = VEL_FILTER_ALPHA * vL_raw + (1.0f - VEL_FILTER_ALPHA) * vL_filt;
        vR_filt = VEL_FILTER_ALPHA * vR_raw + (1.0f - VEL_FILTER_ALPHA) * vR_filt;

        uint32_t elapsed = now - startMs;
        if (elapsed >= (uint32_t)settle_ms)
        {
            sumVL += vL_filt;
            sumVR += vR_filt;
            sampleCount++;
        }

        // Print live reading every ~200ms
        livePrintCounter++;
        if (livePrintCounter >= 10)
        {
            livePrintCounter = 0;
            const char* phase = (elapsed < (uint32_t)settle_ms) ? "SETTLE" : "MEASURE";
            float omega_live = (vR_filt - vL_filt) / TRACK_WIDTH_M;

            char buf[100];
            snprintf(buf, sizeof(buf), "  [%s] vL=%.3f vR=%.3f omega=%.3f\r\n",
                     phase, vL_filt, vR_filt, omega_live);
            out.print(buf);
        }
    }

    setMotorPWMDIR(0, 0);

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
void runMode1(Print& out)
{
    out.println("========================================");
    out.println("MODE 1: SYMMETRIC SWEEP (both motors same PWM)");
    out.println("Purpose: Find FF_PWM_FWD -- the PWM that produces V_FWD");
    out.print("Target V_FWD = ");
    out.print(TARGET_V_FWD, 3);
    out.println(" m/s");
    out.println("========================================");
    out.println(" PWM | vL(m/s)  | vR(m/s)  | vAvg(m/s) | omega(r/s)");
    out.println("-----|----------|----------|-----------|----------");

    int bestPWM = 0;
    float bestErr = 999.0f;

    for (int pwm = SWEEP_MIN; pwm <= SWEEP_MAX; pwm += SWEEP_STEP)
    {
        out.print(">>> Running PWM=");
        out.println(pwm);
        Serial.print("[USB] Mode1 PWM=");
        Serial.println(pwm);

        MeasureResult r = measureSteadyState(out, pwm, pwm);

        char buf[80];
        snprintf(buf, sizeof(buf), " %3d |  %6.3f  |  %6.3f  |   %6.3f   |   %6.3f\r\n",
                 pwm, r.vL, r.vR, r.vAvg, r.omega);
        out.print(buf);

        float err = fabs(r.vAvg - TARGET_V_FWD);
        if (err < bestErr)
        {
            bestErr = err;
            bestPWM = pwm;
        }

        if (fabs(r.vAvg) < 0.005f)
        {
            out.print("  ^ DEADZONE: PWM=");
            out.print(pwm);
            out.println(" produced no motion");
        }

        delay(PAUSE_BETWEEN_MS);
    }

    out.println("========================================");
    out.print("RESULT: For V_FWD=");
    out.print(TARGET_V_FWD, 2);
    out.print(" m/s, best match is FF_PWM_FWD = ");
    out.print(bestPWM);
    out.print(" (vAvg=");
    out.print(TARGET_V_FWD - bestErr, 3);
    out.println(")");
    out.println("========================================");
    out.println();

    Serial.print("[USB] Mode1 complete. Best FF_PWM_FWD = ");
    Serial.println(bestPWM);
}

// ================== MODE 2: DIFFERENTIAL SWEEP ==================
void runMode2(Print& out)
{
    out.println("========================================");
    out.println("MODE 2: DIFFERENTIAL SWEEP (find FF_PWM_OMEGA)");
    out.print("Purpose: Find differential PWM for OMEGA_TURN = ");
    out.print(TARGET_OMEGA, 2);
    out.println(" rad/s");
    out.println("Also tests whether V_TURN > 0 is achievable with a forward bias.");
    out.println("========================================");
    out.println("Base | Diff | cmdL | cmdR | vAvg(m/s) | omega(r/s)");
    out.println("-----|------|------|------|-----------|----------");

    // --- Part A: Pure rotation (base=0) ---
    out.println("--- Pure rotation (base=0) ---");
    Serial.println("[USB] Mode2 Part A: pure rotation");

    int bestDiff_pure = 0;
    float bestOmegaErr_pure = 999.0f;

    for (int diff = 10; diff <= 60; diff += 10)
    {
        int cL = -diff;
        int cR =  diff;
        out.print(">>> Running base=0 diff=");
        out.println(diff);

        MeasureResult r = measureSteadyState(out, cL, cR);

        char buf[80];
        snprintf(buf, sizeof(buf), "   0 |  %3d | %4d | %4d |   %6.3f   |   %6.3f\r\n",
                 diff, cL, cR, r.vAvg, r.omega);
        out.print(buf);

        float omegaErr = fabs(fabs(r.omega) - TARGET_OMEGA);
        if (omegaErr < bestOmegaErr_pure)
        {
            bestOmegaErr_pure = omegaErr;
            bestDiff_pure = diff;
        }

        delay(PAUSE_BETWEEN_MS);
    }

    // --- Part B: With forward bias (base=40) ---
    out.println("--- With forward bias (base=40) ---");
    Serial.println("[USB] Mode2 Part B: rotation + forward bias");

    for (int diff = 10; diff <= 50; diff += 10)
    {
        int base = 40;
        int cL = base - diff;
        int cR = base + diff;
        out.print(">>> Running base=40 diff=");
        out.println(diff);

        MeasureResult r = measureSteadyState(out, cL, cR);

        char buf[80];
        snprintf(buf, sizeof(buf), "  40 |  %3d | %4d | %4d |   %6.3f   |   %6.3f\r\n",
                 diff, cL, cR, r.vAvg, r.omega);
        out.print(buf);

        delay(PAUSE_BETWEEN_MS);
    }

    out.println("========================================");
    out.print("RESULT: For pure rotation at OMEGA=");
    out.print(TARGET_OMEGA, 2);
    out.print(" rad/s, best match is FF_PWM_OMEGA = ");
    out.println(bestDiff_pure);
    out.println("Check base=40 rows: if vAvg > 0.02, V_TURN > 0 is achievable.");
    out.println("If vAvg ~ 0 for all base=40 rows, set V_TURN=0 and FF_PWM_TURN=0.");
    out.println("========================================");
    out.println();

    Serial.print("[USB] Mode2 complete. Best FF_PWM_OMEGA = ");
    Serial.println(bestDiff_pure);
}

// ================== MODE 3: TURN PROFILE VALIDATION ==================
void runMode3(Print& out)
{
    out.println("========================================");
    out.println("MODE 3: TURN PROFILE VALIDATION");
    out.print("FF values: FWD=");
    out.print(FF_PWM_FWD);
    out.print(" TURN=");
    out.print(FF_PWM_TURN);
    out.print(" OMEGA=");
    out.println(FF_PWM_OMEGA);
    out.println("========================================");
    out.println("Action   | cmdL | cmdR | vAvg(m/s) | omega(r/s) | tgt_v | tgt_w | err_v | err_w");
    out.println("---------|------|------|-----------|------------|-------|-------|-------|------");

    struct ActionDef
    {
        const char* name;
        int cmdL, cmdR;
        float tgt_v, tgt_w;
    };

    ActionDef actions[3] = {
        {"FORWARD ", FF_PWM_FWD,                FF_PWM_FWD,                TARGET_V_FWD,  0.0f},
        {"LEFT    ", FF_PWM_TURN - FF_PWM_OMEGA, FF_PWM_TURN + FF_PWM_OMEGA, TARGET_V_TURN,  TARGET_OMEGA},
        {"RIGHT   ", FF_PWM_TURN + FF_PWM_OMEGA, FF_PWM_TURN - FF_PWM_OMEGA, TARGET_V_TURN, -TARGET_OMEGA}
    };

    for (int i = 0; i < 3; i++)
    {
        ActionDef& a = actions[i];
        out.print(">>> Running ");
        out.println(a.name);
        Serial.print("[USB] Mode3: ");
        Serial.println(a.name);

        MeasureResult r = measureSteadyState(out, a.cmdL, a.cmdR);

        float err_v = r.vAvg - a.tgt_v;
        float err_w = r.omega - a.tgt_w;

        char buf[120];
        snprintf(buf, sizeof(buf),
                 "%s | %4d | %4d |   %6.3f  |    %6.3f  |  %4.2f | %4.2f | %+5.3f | %+5.3f\r\n",
                 a.name, a.cmdL, a.cmdR, r.vAvg, r.omega, a.tgt_v, a.tgt_w, err_v, err_w);
        out.print(buf);

        bool pass_v = fabs(err_v) < 0.05f;
        bool pass_w = fabs(err_w) < 0.15f;
        out.print("  -> v: ");
        out.print(pass_v ? "PASS" : "FAIL");
        out.print("  w: ");
        out.println(pass_w ? "PASS" : "FAIL");

        delay(PAUSE_BETWEEN_MS);
    }

    out.println("========================================");
    out.println("If FAIL: adjust FF values at top of file and re-upload.");
    out.println("========================================");

    Serial.println("[USB] Mode3 complete.");
}

// ================== SETUP & LOOP ==================
void setup()
{

    pwm_L.period_us(2000); 
    pwm_R.period_us(2000);
    // ========================================================
    // 1. IMMEDIATELY LOCK DOWN MOTORS TO PREVENT BOOT SPIN
    // ========================================================
    // No pinMode needed for Mbed objects. Just shut off the duty cycle.
    pwm_L.write(0.0f);
    pwm_R.write(0.0f);

    // USB serial for debug mirroring
    Serial.begin(115200);
    while (!Serial && millis() < 3000)
        ;

    // Ethernet init with static IP
    Ethernet.begin(mac, ip);
    server.begin();

    // Encoder pins
    encL_A.mode(PullUp);
    encL_B.mode(PullUp);
    encR_A.mode(PullUp);
    encR_B.mode(PullUp);

    encL_A.rise(&isrEncL_A);
    encL_A.fall(&isrEncL_A);
    encR_A.rise(&isrEncR_A);
    encR_A.fall(&isrEncR_A);

    Serial.println("[USB] ETH calibration ready. Waiting for TCP client...");
    Serial.print("[USB] Connect to ");
    Serial.print(ip);
    Serial.print(":");
    Serial.println(ETH_PORT);
}

void loop()
{
    // Accept new client if we don't have one connected
    if (!client || !client.connected())
    {
        client = server.available();
        welcomeSent = false;
        if (!client)
            return;
        Serial.println("[USB] TCP client connected.");
    }

    // Send welcome message once per connection
    if (!welcomeSent)
    {
        welcomeSent = true;
        client.println("========================================");
        client.println("ETH FEEDFORWARD PWM CALIBRATION TEST");
        client.println("!! ELEVATE ROBOT -- wheels must be OFF the ground !!");
        client.print("PWM_LIMIT = ");
        client.print(PWM_LIMIT);
        client.print("  VEL_FILTER_ALPHA = ");
        client.println(VEL_FILTER_ALPHA, 1);
        client.println("Send any character to begin...");
        client.println("========================================");
    }

    // Wait for any character from the TCP client
    if (!client.available())
        return;

    // Drain any extra bytes (e.g. CR+LF from terminal)
    while (client.available())
        client.read();

    Serial.println("[USB] Trigger received. Starting calibration in 5s...");
    client.println();
    client.println("Trigger received! Starting in 5 seconds...");
    client.println("========================================");
    delay(5000);

    // runMode1(client);
    // delay(3000);
    // runMode2(client);
    // delay(3000);
    runMode3(client);

    client.println();
    client.println("========================================");
    client.println("CALIBRATION COMPLETE");
    client.println();
    client.println("Next steps:");
    client.println("  1. Note the best FF_PWM_FWD from Mode 1");
    client.println("  2. Note the best FF_PWM_OMEGA from Mode 2");
    client.println("  3. Check Mode 2 base=40 rows to decide V_TURN");
    client.println("  4. Update FF values at top of this file");
    client.println("  5. Re-upload and send any character to re-trigger");
    client.println("  6. Copy final values to chassis.cpp");
    client.println("========================================");

    Serial.println("[USB] Calibration complete.");

    // Sit idle — re-upload to run again, or send another character to re-trigger
    // (loop() will fire again if another character arrives)
}
