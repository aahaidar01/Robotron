/*
 * TEST: Motors + Encoders
 * ========================
 * Verifies motor PWM/DIR wiring, rotation direction, and RL action mapping.
 * Uses EXACT same pin definitions, motor control, and encoder ISRs as firmware.
 * Runs a sequence of motor commands with encoder feedback.
 *
 * ⚠️  SAFETY: ELEVATE THE ROBOT so wheels are OFF the ground!
 *
 * HOW TO USE:
 *   1. Put robot on a box (wheels in the air)
 *   2. Upload to Portenta H7
 *   3. Open Serial Monitor at 115200
 *   4. Robot will wait 5 seconds, then run through test phases:
 *
 *   Phase 1: Left motor FWD   (3s) — LEFT wheels spin forward?
 *   Phase 2: Left motor REV   (3s) — LEFT wheels spin backward?
 *   Phase 3: Right motor FWD  (3s) — RIGHT wheels spin forward?
 *   Phase 4: Right motor REV  (3s) — RIGHT wheels spin backward?
 *   Phase 5: RL FORWARD       (3s) — both motors forward?
 *   Phase 6: RL LEFT TURN     (3s) — robot would turn left (CCW)?
 *   Phase 7: RL RIGHT TURN    (3s) — robot would turn right (CW)?
 *   Phase 8: STOP             — all motors off, test complete
 *
 * IF WRONG MOTOR RESPONDS:
 *   Swap PWM_L/PWM_R and DIR_L/DIR_R pin assignments
 *
 * IF DIRECTION IS BACKWARDS:
 *   Swap HIGH/LOW in digitalWrite(DIR_x, ...) for that motor
 */

#include "mbed.h"
#include <Arduino.h>

// ---- Motor Pins (SAME as chassis.cpp) ----
const int PWM_L = 6;
const int DIR_L = 5;
const int PWM_R = 2;
const int DIR_R = 1;

// ---- Encoder Pins (SAME as chassis.cpp) ----
// Right Encoder
mbed::InterruptIn encR_A(PD_4);
mbed::DigitalIn   encR_B(PG_10);

// Left Encoder (Using standard Arduino pins 4 and 0 via Mbed)
mbed::InterruptIn encL_A(PC_7);  // Arduino Pin 4 resolves to PC_7 on Portenta
mbed::DigitalIn   encL_B(PH_15); // Arduino Pin 0 resolves to PH_15 on Portenta

// ---- Constants (SAME as chassis.cpp) ----
static constexpr int PWM_LIMIT = 191;
static constexpr float CPR_MOTOR = 48.0f;
static constexpr float GEAR_RATIO_EXACT = 34.014f;
static constexpr float CPR_WHEEL = CPR_MOTOR * GEAR_RATIO_EXACT;
static constexpr float WHEEL_DIAM_M = 0.12f;
static constexpr float WHEEL_CIRC_M = 3.1415926f * WHEEL_DIAM_M;
static constexpr float METERS_PER_COUNT = WHEEL_CIRC_M / CPR_WHEEL;

// ---- Encoder ISRs (SAME as chassis.cpp) ----
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

// ---- Motor Control (SAME as chassis.cpp) ----
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

// ---- Test Helpers ----
const int TEST_PWM = 60; // Low PWM for safe testing

void runPhase(const char* name, int cmdL, int cmdR, int duration_ms)
{
    Serial.println("------------------------------------------------");
    Serial.print(">>> "); Serial.print(name);
    Serial.print(" | cmdL="); Serial.print(cmdL);
    Serial.print(" cmdR="); Serial.println(cmdR);

    // Reset tick counters for this phase
    noInterrupts();
    int32_t startL = ticksL;
    int32_t startR = ticksR;
    interrupts();

    setMotorPWMDIR(cmdL, cmdR);

    // Print encoder feedback during the phase
    uint32_t start = millis();
    while (millis() - start < (uint32_t)duration_ms)
    {
        delay(300);
        noInterrupts();
        int32_t tL = ticksL;
        int32_t tR = ticksR;
        interrupts();

        Serial.print("  ticks: L=");
        Serial.print(tL - startL);
        Serial.print(" R=");
        Serial.println(tR - startR);
    }

    // Stop and show totals
    setMotorPWMDIR(0, 0);

    noInterrupts();
    int32_t endL = ticksL;
    int32_t endR = ticksR;
    interrupts();

    int32_t dL = endL - startL;
    int32_t dR = endR - startR;
    float distL = dL * METERS_PER_COUNT;
    float distR = dR * METERS_PER_COUNT;

    Serial.print("  RESULT: deltaL="); Serial.print(dL);
    Serial.print(" ("); Serial.print(distL * 100, 1); Serial.print("cm)");
    Serial.print(" | deltaR="); Serial.print(dR);
    Serial.print(" ("); Serial.print(distR * 100, 1); Serial.println("cm)");
    Serial.println();

    delay(1000); // Pause between phases
}

void setup()
{
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

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

    // Attach Interupts for both RISE and FALL to mimic Arduino's 'CHANGE'
    encL_A.rise(&isrEncL_A);
    encL_A.fall(&isrEncL_A);
    
    encR_A.rise(&isrEncR_A);
    encR_A.fall(&isrEncR_A);

    Serial.println("================================================");
    Serial.println("MOTOR TEST");
    Serial.println("!! ELEVATE ROBOT — wheels must be OFF the ground !!");
    Serial.print("Using PWM = "); Serial.print(TEST_PWM);
    Serial.print(" out of "); Serial.println(PWM_LIMIT);
    Serial.println("Starting in 5 seconds...");
    Serial.println("================================================");
    delay(5000);

    // ---- Phase 1-4: Individual motor tests ----
    runPhase("LEFT motor FORWARD",   TEST_PWM, 0, 3000);
    runPhase("LEFT motor REVERSE",  -TEST_PWM, 0, 3000);
    runPhase("RIGHT motor FORWARD",  0, TEST_PWM, 3000);
    runPhase("RIGHT motor REVERSE",  0, -TEST_PWM, 3000);

    // ---- Phase 5-7: RL action tests ----
    // These simulate the 3 discrete actions from the Q-table
    runPhase("RL Action 0: FORWARD (both fwd)",
             TEST_PWM, TEST_PWM, 3000);
    runPhase("RL Action 1: LEFT TURN (right faster, left slower)",
             TEST_PWM / 2, TEST_PWM, 3000);
    runPhase("RL Action 2: RIGHT TURN (left faster, right slower)",
             TEST_PWM, TEST_PWM / 2, 3000);

    // ---- Done ----
    setMotorPWMDIR(0, 0);
    Serial.println("================================================");
    Serial.println("MOTOR TEST COMPLETE");
    Serial.println("");
    Serial.println("Check the results above:");
    Serial.println("  - Did LEFT/RIGHT motors match expectations?");
    Serial.println("  - Did forward produce POSITIVE encoder ticks?");
    Serial.println("  - Did RL actions produce correct turn directions?");
    Serial.println("================================================");
}

void loop()
{
    // Nothing — test runs once in setup()
}
