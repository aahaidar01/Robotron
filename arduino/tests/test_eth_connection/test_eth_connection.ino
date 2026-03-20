/*
 * ETHERNET CONNECTION TEST
 * ========================
 * Verifies the Ethernet link between Portenta H7 and your PC
 * before running motor calibration. No motors or encoders involved.
 *
 * HOW TO USE:
 *   1. Connect Ethernet cable between Portenta and PC.
 *   2. Set your PC's Ethernet adapter to static IP 192.168.1.100,
 *      netmask 255.255.255.0.
 *   3. Upload this sketch to Portenta H7.
 *   4. Open USB serial monitor at 115200 to see status.
 *   5. From your PC terminal: nc 192.168.1.177 23
 *   6. Type anything — Portenta echoes it back and sends a heartbeat
 *      every 2 seconds to confirm the link is alive.
 */

#include <Arduino.h>
#include <PortentaEthernet.h>
#include <Ethernet.h>

byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
IPAddress ip(192, 168, 1, 177);
const int ETH_PORT = 23;

EthernetServer server(ETH_PORT);
EthernetClient client;

static uint32_t heartbeatCount = 0;
static uint32_t lastHeartbeatMs = 0;
static const uint32_t HEARTBEAT_INTERVAL_MS = 2000;

void setup()
{
    Serial.begin(115200);
    while (!Serial && millis() < 3000)
        ;

    Serial.println("[USB] Initializing Ethernet...");
    Ethernet.begin(mac, ip);
    server.begin();

    Serial.print("[USB] Server ready at ");
    Serial.print(ip);
    Serial.print(":");
    Serial.println(ETH_PORT);
    Serial.println("[USB] Waiting for TCP client...");
}

void loop()
{
    if (!client || !client.connected())
    {
        client = server.available();
        if (!client)
            return;

        Serial.println("[USB] Client connected!");
        heartbeatCount = 0;
        lastHeartbeatMs = millis();

        client.println("========================================");
        client.println("ETHERNET CONNECTION TEST — OK");
        client.print("Portenta IP: ");
        client.println(ip);
        client.println("Type anything to echo. Heartbeat every 2s.");
        client.println("========================================");
    }

    // Echo back any received bytes
    while (client.available())
    {
        char c = client.read();
        client.print("echo> ");
        client.println(c);
        Serial.print("[USB] Received: ");
        Serial.println(c);
    }

    // Periodic heartbeat
    uint32_t now = millis();
    if (now - lastHeartbeatMs >= HEARTBEAT_INTERVAL_MS)
    {
        lastHeartbeatMs = now;
        heartbeatCount++;

        char buf[60];
        snprintf(buf, sizeof(buf), "heartbeat #%lu  uptime=%lus",
                 (unsigned long)heartbeatCount,
                 (unsigned long)(now / 1000));
        client.println(buf);
        Serial.print("[USB] ");
        Serial.println(buf);
    }
}
