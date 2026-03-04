#ifndef CHASSIS_H
#define CHASSIS_H

void init_chassis();
void update_chassis(); // Runs the 50Hz PID loop

void execute_motor_command(int action);
void emergency_stop();

// Odometry getter for the Lidar math
void get_current_pose(float &x, float &y, float &yaw);

// Terminal condition check
bool is_target_reached();


void reset_odometry();

#endif