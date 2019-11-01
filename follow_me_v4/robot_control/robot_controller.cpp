#include "robot_controller.h"

RobotController::RobotController()
{
	// initialize PID controllers
	turn_rate_pid.set_coefficients(-0.65, 0, -0.02);
	//velocity_pid.set_coefficients(15, 0, 0.4);
	velocity_pid.set_coefficients(0, 0, 0); // TODO deal with linear velocity
}

// control the robot as a function of perpendicular distance (meters)
// and the object's centroid in frame (pixels, relative to frame center)
void RobotController::control(double perp_distance, int centroid_x, int centroid_y)
{
	double vel = velocity_pid.control(perp_distance - 2.0);
	double turn_rate = turn_rate_pid.control(centroid_x);
	serial_comm.write_velocity(vel, turn_rate);
}

// control velocity and turn rate directly
void RobotController::control(int vel, int turn_rate)
{
	serial_comm.write_velocity(vel, turn_rate);
}
