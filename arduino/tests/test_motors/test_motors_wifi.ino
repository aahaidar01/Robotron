/*
 * WIFI AP FEEDFORWARD PWM CALIBRATION TEST
 * ==================================
 * Maps PWM values to measured wheel velocities so you can set
 * FF_PWM_FWD, FF_PWM_TURN, and FF_PWM_OMEGA in chassis.cpp.
 *
 * Uses the SAME encoder constants, ISRs, pin definitions, velocity
 * filter (alpha=0.3), and PWM_LIMIT (120) as the main firmware.
 *
 * SAFETY: ELEVATE THE ROBOT — wheels must be OFF the ground!
 *
 * HOW TO USE:
 * 1. Elevate robot.
 * 2. Upload this script to the Portenta H7.
 * 3. Connect your computer's Wi-Fi to the "Portenta_Calibration" network.
 * 4. Open USB Serial Monitor (115200 baud) to double-check the IP address.
 * 5. Connect via Telnet (Port 23) to that IP. The test will start automatically.
 * 6. Read the tables, update the FF constants below, and re-upload to validate.
 */

#include "mbed.h"
#include <Arduino.h>
#include <WiFi.h>

// ================== NETWORK CONFIGURATION ==================
// The Portenta will CREATE this network.
const char* ssid = "Portenta_Calibration";
const char* password = "traverseML1234"; // Must be at least 8 characters
WiFiServer server(23); // Telnet server port

// ================== USER CONFIGURATION ==================
// Set these to your best-guess FF values. After running Mode 1 & 2,
// update them and re-upload to validate with Mode 3.
static int FF_PWM_FWD   = 55;   // PWM for ~V_FWD (0.18 m/s)
static int FF_PWM_TURN  = 40;   // Symmetric speed during turns
static int FF_PWM_OMEGA = 40;   // Differential for ~OMEGA_TURN (0.5 rad/s)

// Target velocities (must match simulation action definitions)
static constexpr float TARGET_V_FWD  = 0.18f;   // m/s
static constexpr float TARGET_V_TURN = 0.03f;   // m/s
static constexpr float TARGET_OMEGA  = 0.50f;   // rad/s

// Sweep configuration
static constexpr int SWEEP_MIN  = 20;
static constexpr int SWEEP_MAX  = 120;
static constexpr int SWEEP_STEP = 10;

// Timing (ms)
static constexpr int RUN_DURATION_MS  = 3000; // Total per step
static constexpr int SETTLE_MS        = 1500; // Discard initial transient
static constexpr int PAUSE_BETWEEN_MS = 1000; // Motor cool-down between steps

// ================== HARDWARE CONFIG ==================
const int PWM_L = 6;
const int DIR_L = 5;
const int PWM_R = 2;
const int DIR_R = 1;

// Encoder pins
mbed::InterruptIn encR_A(PD_4);
mbed::DigitalIn   encR_B(PG_10);
mbed::InterruptIn encL_A(PC_7);
mbed::DigitalIn   encL_B(PH_15);

// Constants
static constexpr int   PWM_LIMIT        = 120;
static constexpr float TRACK_WIDTH_M    = 0.26f;
static constexpr float CPR_MOTOR        = 48.0f;
static constexpr float GEAR_RATIO_EXACT = 34.014f;
static constexpr float CPR_WHEEL        = CPR_MOTOR * GEAR_RATIO_EXACT;
static constexpr float WHEEL_DIAM_M     = 0.12f;
static constexpr float WHEEL_CIRC_M     = 3.1415926f * WHEEL_DIAM_M;
static constexpr float METERS_PER_COUNT = WHEEL_CIRC_M / CPR_WHEEL;
static constexpr float VEL_FILTER_ALPHA = 0.3f;
static constexpr uint32_t SAMPLE_DT_MS  = 20; // 50Hz

// ================== ENCODER ISRs ==================
volatile int32_t ticksL = 0;
volatile int32_t ticksR = 0;

void isrEncL_A() {
    bool a = encL_A.read();
    bool b = encL_B.read();
    ticksL += (a == b) ? +1 : -1;
}

void isrEncR_A() {
    bool a = encR_A.read();
    bool b = encR_B.read();
    ticksR += (a == b) ? +1 : -1;
}

// ================== MOTOR CONTROL ==================
static inline int clampMag(int mag) { return (mag > PWM_LIMIT) ? PWM_LIMIT : mag; }

void setMotorPWMDIR(int cmdL, int cmdR) {
    int magL = clampMag(abs(cmdL));
    int magR = clampMag(abs(cmdR));
    digitalWrite(DIR_L, (cmdL >= 0) ? HIGH : LOW);
    digitalWrite(DIR_R, (cmdR >= 0) ? HIGH : LOW);
    analogWrite(PWM_L, magL);
    analogWrite(PWM_R, magR);
}

// ================== MEASUREMENT ==================
struct MeasureResult {
    float vL;     // Filtered left velocity (m/s)
    float vR;     // Filtered right velocity (m/s)
    float vAvg;   // Average linear velocity (m/s)
    float omega;  // Yaw rate from encoders (rad/s)
};

MeasureResult measureSteadyState(Print& out, int cmdL, int cmdR, int total_ms = RUN_DURATION_MS, int settle_ms = SETTLE_MS) {
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

    while (millis() - startMs < (uint32_t)total_ms) {
        uint32_t now = millis();
        if (now - lastSampleMs < SAMPLE_DT_MS) continue;
        
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
        if (elapsed >= (uint32_t)settle_ms) {
            sumVL += vL_filt;
            sumVR += vR_filt;
            sampleCount++;
        }

        livePrintCounter++;
        if (livePrintCounter >= 10) { // 10 * 20ms = 200ms
            livePrintCounter = 0;
            const char* phase = (elapsed < (uint32_t)settle_ms) ? "SETTLE" : "MEASURE";
            float omega_live = (vR_filt - vL_filt) / TRACK_WIDTH_M;
            
            // Buffer the output to prevent Wi-Fi lag overhead
            char buf[100];
            snprintf(buf, sizeof(buf), "  [%s] vL=%.3f vR=%.3f omega=%.3f\r\n", phase, vL_filt, vR_filt, omega_live);
            out.print(buf);
        }
    }

    setMotorPWMDIR(0, 0);

    MeasureResult r;
    if (sampleCount > 0) {
        r.vL = sumVL / sampleCount;
        r.vR = sumVR / sampleCount;
    } else {
        r.vL = 0.0f;
        r.vR = 0.0f;
    }
    r.vAvg = 0.5f * (r.vL + r.vR);
    r.omega = (r.vR - r.vL) / TRACK_WIDTH_M;
    return r;
}

// ================== MODE 1: SYMMETRIC SWEEP ==================
void runMode1(Print& out) {
    out.println("========================================");
    out.println("MODE 1: SYMMETRIC SWEEP (both motors same PWM)");
    out.println("Purpose: Find FF_PWM_FWD — the PWM that produces V_FWD");
    out.print("Target V_FWD = ");
    out.print(TARGET_V_FWD, 3);
    out.println(" m/s");
    out.println("========================================");
    out.println(" PWM | vL(m/s)  | vR(m/s)  | vAvg(m/s) | omega(r/s)");
    out.println("-----|----------|----------|-----------|----------");

    int bestPWM = 0;
    float bestErr = 999.0f;

    for (int pwm = SWEEP_MIN; pwm <= SWEEP_MAX; pwm += SWEEP_STEP) {
        out.print(">>> Running PWM=");
        out.println(pwm);

        MeasureResult r = measureSteadyState(out, pwm, pwm);

        char buf[80];
        snprintf(buf, sizeof(buf), " %3d |  %6.3f  |  %6.3f  |   %6.3f   |   %6.3f\r\n", 
                 pwm, r.vL, r.vR, r.vAvg, r.omega);
        out.print(buf);

        float err = fabs(r.vAvg - TARGET_V_FWD);
        if (err < bestErr) {
            bestErr = err;
            bestPWM = pwm;
        }

        if (fabs(r.vAvg) < 0.005f) {
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
}

// ================== MODE 2: DIFFERENTIAL SWEEP ==================
void runMode2(Print& out) {
    out.println("========================================");
    out.println("MODE 2: DIFFERENTIAL SWEEP (find FF_PWM_OMEGA)");
    out.print("Purpose: Find differential PWM for OMEGA_TURN = ");
    out.print(TARGET_OMEGA, 2);
    out.println(" rad/s");
    out.println("Also tests whether V_TURN > 0 is achievable with a forward bias.");
    out.println("========================================");
    out.println("Base | Diff | cmdL | cmdR | vAvg(m/s) | omega(r/s)");
    out.println("-----|------|------|------|-----------|----------");
    
    out.println("--- Pure rotation (base=0) ---");
    int bestDiff_pure = 0;
    float bestOmegaErr_pure = 999.0f;

    for (int diff = 10; diff <= 60; diff += 10) {
        int cL = -diff;
        int cR = diff;
        out.print(">>> Running base=0 diff=");
        out.println(diff);

        MeasureResult r = measureSteadyState(out, cL, cR);
        
        char buf[80];
        snprintf(buf, sizeof(buf), "   0 |  %3d | %4d | %4d |   %6.3f   |   %6.3f\r\n", 
                 diff, cL, cR, r.vAvg, r.omega);
        out.print(buf);

        float omegaErr = fabs(fabs(r.omega) - TARGET_OMEGA);
        if (omegaErr < bestOmegaErr_pure) {
            bestOmegaErr_pure = omegaErr;
            bestDiff_pure = diff;
        }
        delay(PAUSE_BETWEEN_MS);
    }

    out.println("--- With forward bias (base=40) ---");
    for (int diff = 10; diff <= 50; diff += 10) {
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
}

// ================== MODE 3: TURN PROFILE Validation ==================
void runMode3(Print& out) {
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

    struct ActionDef {
        const char* name;
        int cmdL, cmdR;
        float tgt_v, tgt_w;
    };

    ActionDef actions[3] = {
        {"FORWARD ", FF_PWM_FWD, FF_PWM_FWD, TARGET_V_FWD, 0.0f},
        {"LEFT    ", FF_PWM_TURN - FF_PWM_OMEGA, FF_PWM_TURN + FF_PWM_OMEGA, TARGET_V_TURN, TARGET_OMEGA},
        {"RIGHT   ", FF_PWM_TURN + FF_PWM_OMEGA, FF_PWM_TURN - FF_PWM_OMEGA, TARGET_V_TURN, -TARGET_OMEGA}
    };

    for (int i = 0; i < 3; i++) {
        ActionDef& a = actions[i];
        out.print(">>> Running ");
        out.println(a.name);

        MeasureResult r = measureSteadyState(out, a.cmdL, a.cmdR);

        float err_v = r.vAvg - a.tgt_v;
        float err_w = r.omega - a.tgt_w;

        char buf[120];
        snprintf(buf, sizeof(buf), "%s | %4d | %4d |   %6.3f  |    %6.3f  |  %4.2f | %4.2f | %+5.3f | %+5.3f\r\n",
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
}

// ================== SETUP & LOOP ==================
void setup() {
    Serial.begin(115200);
    
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

    // Create Wi-Fi Access Point
    Serial.println("Creating Access Point...");
    int status = WiFi.beginAP(ssid, password);
    if (status != WL_AP_LISTENING) {
        Serial.println("Creating Access Point failed. Halting.");
        while (true); // Freeze if it fails
    }

    delay(2000); // Give the AP a moment to stabilize
    
    Serial.println("\nAccess Point Created!");
    Serial.print("Network Name (SSID): ");
    Serial.println(ssid);
    Serial.print("Password: ");
    Serial.println(password);
    Serial.print("Connect your computer to this Wi-Fi, then Telnet to IP: ");
    Serial.println(WiFi.localIP());
    Serial.println("Port: 23");

    server.begin();
}

void loop() {
    WiFiClient client = server.available();
    
    // When a terminal connects, start the sequence
    if (client) {
        client.println("========================================");
        client.println("WIFI AP FEEDFORWARD PWM CALIBRATION TEST");
        client.println("!! ELEVATE ROBOT — wheels must be OFF the ground !!");
        client.print("PWM_LIMIT = ");
        client.print(PWM_LIMIT);
        client.print("  VEL_FILTER_ALPHA = ");
        client.println(VEL_FILTER_ALPHA, 1);
        client.println("Starting in 5 seconds...");
        client.println("========================================");
        delay(5000);

        // Pass the 'client' object as the Print reference
        runMode1(client);
        delay(3000);
        runMode2(client);
        delay(3000);
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
        client.println("  5. Re-upload and check Mode 3 passes");
        client.println("  6. Copy final values to chassis.cpp");
        client.println("========================================");
        
        // Drop the connection so we don't repeat the test endlessly
        client.stop(); 
    }
}