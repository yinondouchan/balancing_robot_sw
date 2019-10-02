#include "robot_controller.h"

#include <iostream>

RobotController::RobotController()
{
	// initialize PID controllers
	turn_rate_pid.set_coefficients(0.75, 0, 0.02);
	//turn_rate_pid.set_p_lpf(0.01);
	//turn_rate_pid.set_d_lpf(0.01);
	velocity_pid.set_coefficients(-0.009, 0, 0.0003);
}

// control the robot as a function of perpendicular distance (meters)
// and the object's centroid in frame (pixels, relative to frame center)
void RobotController::control(double perp_distance, int centroid_x, int centroid_y)
{
	double vel = velocity_pid.control(perp_distance - 2.0);
	double turn_rate = turn_rate_pid.control(centroid_x);
	serial_comm.write_velocity(vel, turn_rate);
}

// control the robot as a function of detection area (pixels^2)
// and the object's centroid in frame (pixels, relative to frame center)
void RobotController::control_by_area_and_centroid(double bbox_area, int centroid_x, int centroid_y)
{
	double vel = velocity_pid.control(bbox_area - 100000.0);
	double turn_rate = turn_rate_pid.control(centroid_x);
	serial_comm.write_velocity(vel, turn_rate);
}

// control velocity and turn rate directly
void RobotController::control(int vel, int turn_rate)
{
	serial_comm.write_velocity(vel, turn_rate);
}
