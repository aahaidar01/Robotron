/*
 * TEST: Encoders
 * ===============
 * Verifies encoder wiring, interrupt firing, and quadrature direction.
 */

#include "mbed.h"
#include <Arduino.h>

// ---- Geometry ----
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

// ---- Mbed Pin Objects ----
// Create Interrupt Objects
// Hat Carrier 40-pin header assignments (matched to comments):
// Left Encoder
mbed::InterruptIn encL_A(PD_4);
mbed::DigitalIn   encL_B(PG_10);

// Right Encoder (Using standard Arduino pins 4 and 0 via Mbed)
mbed::InterruptIn encR_A(PC_7);  // Arduino Pin 4 resolves to PC_7 on Portenta
mbed::DigitalIn   encR_B(PH_15); // Arduino Pin 0 resolves to PH_15 on Portenta

// ---- ISRs ----
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

void setup()
{
    Serial.begin(115200);

    // Configure pullups using Mbed API
    encL_A.mode(PullUp);
    encL_B.mode(PullUp);
    encR_A.mode(PullUp);
    encR_B.mode(PullUp);

    // Attach interrupts for both RISE and FALL to mimic Arduino's 'CHANGE'
    encL_A.rise(&isrEncL_A);
    encL_A.fall(&isrEncL_A);
    
    encR_A.rise(&isrEncR_A);
    encR_A.fall(&isrEncR_A);

    Serial.println("================================================");
    Serial.println("ENCODER TEST — Push wheels by hand");
    Serial.println("Forward = ticks should INCREASE");
    Serial.println("================================================");
}

void loop()
{
    static uint32_t lastMs = 0;
    uint32_t now = millis();
    if (now - lastMs < 200) return; 

    float dt = (float)(now - lastMs) / 1000.0f;
    lastMs = now;

    // Safely copy volatile variables
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