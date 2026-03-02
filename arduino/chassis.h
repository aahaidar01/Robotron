#ifndef CHASSIS_H
#define CHASSIS_H

// Instantly cuts power to the motors. Bypasses the RL loop entirely.
void emergency_stop();

// Allows the LiDAR code to safely read the odometry without global variables
void get_current_pose(float &x, float &y, float &yaw);

#endif