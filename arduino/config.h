// config.h
#ifndef CONFIG_H
#define CONFIG_H

// Target location in world frame (change per maze layout)
const float TARGET_X = 0.0f;
const float TARGET_Y = 1.8f;
const float TARGET_RADIUS = 0.30f;

// Robot starting position in world frame (must match physical placement)
// These offset the encoder odometry so all distance/angle calculations
// use the same coordinate system as simulation training.
//
// Common spawns from training (copy the one matching your physical setup):
//   Easy:   (0.5, 1.3, 1.57)   — Zone 3, facing north
//   Medium: (-0.5, 0.0, 1.57)  — Zone 2, facing north
//   Hard:   (-0.5, -2.1, 0.0)  — Start zone, facing east (+X)
const float SPAWN_X   = -0.5f;
const float SPAWN_Y   = -2.1f;
const float SPAWN_YAW =  0.0f;   // radians, 0 = facing +X // we have to confirm with the IMU convention by testing

#endif