/*
 * TEST: Encoders
 * ===============
 * Verifies encoder wiring, interrupt firing, and quadrature direction.
 * Uses EXACT same pin definitions and ISR logic as main firmware.
 *
 * HOW TO USE:
 *   1. Upload to Portenta H7
 *   2. Open Serial Monitor at 115200
 *   3. Push LEFT wheels forward by hand  → ticks_L should INCREASE
 *   4. Push RIGHT wheels forward by hand → ticks_R should INCREASE
 *   5. Speed should read ~0.1-0.3 m/s when pushed at walking pace
 *
 * IF DIRECTION IS WRONG:
 *   - Swap Ch A and Ch B wires for that encoder
 *   - OR flip the sign in the ISR: change +1/-1 to -1/+1
 */

#include "mbed.h"
#include <Arduino.h>

// ---- Pin Definitions (SAME as chassis.cpp) ----
// Hat Carrier 40-pin header:
//   GPIO2 (pin 11) -> PD_4   -> Encoder L Channel A
//   GPIO6 (pin 13) -> PG_10  -> Encoder L Channel B
//   PWM2  (pin 26) -> Arduino 4 (PC_7)   -> Encoder R Channel A
//   PWM6  (pin 37) -> Arduino 0 (PH_15)  -> Encoder R Channel B
const int ENC_L_A = PD_4;
const int ENC_L_B = PG_10;
const int ENC_R_A = 4;
const int ENC_R_B = 0;

// ---- Geometry (SAME as chassis.cpp) ----
static constexpr float CPR_MOTOR = 48.0f;
static constexpr float GEAR_RATIO_EXACT = 34.014f;
static constexpr float CPR_WHEEL = CPR_MOTOR * GEAR_RATIO_EXACT;
static constexpr float WHEEL_DIAM_M = 0.12f;
static constexpr float WHEEL_CIRC_M = 3.1415926f * WHEEL_DIAM_M;
static constexpr float METERS_PER_COUNT = WHEEL_CIRC_M / CPR_WHEEL;

// ---- Tick Counters ----
volatile int32_t ticksL = 0;
volatile int32_t ticksR = 0;
int32_t prevTicksL = 0, prevTicksR = 0;

// ---- ISRs (SAME as chassis.cpp) ----
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

void setup()
{
    Serial.begin(115200);
    while (!Serial && millis() < 3000); // Wait for Serial (up to 3s)

    pinMode(ENC_L_A, INPUT_PULLUP);
    pinMode(ENC_L_B, INPUT_PULLUP);
    pinMode(ENC_R_A, INPUT_PULLUP);
    pinMode(ENC_R_B, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrEncL_A, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrEncR_A, CHANGE);

    Serial.println("================================================");
    Serial.println("ENCODER TEST — Push wheels by hand");
    Serial.println("Forward = ticks should INCREASE");
    Serial.println("================================================");
}

void loop()
{
    static uint32_t lastMs = 0;
    uint32_t now = millis();
    if (now - lastMs < 200) return; // Print every 200ms

    float dt = (float)(now - lastMs) / 1000.0f;
    lastMs = now;

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

    Serial.print("ticks: L=");
    Serial.print(tL);
    Serial.print(" R=");
    Serial.print(tR);
    Serial.print(" | delta: L=");
    Serial.print(dL);
    Serial.print(" R=");
    Serial.print(dR);
    Serial.print(" | speed(m/s): L=");
    Serial.print(vL, 3);
    Serial.print(" R=");
    Serial.println(vR, 3);
}
