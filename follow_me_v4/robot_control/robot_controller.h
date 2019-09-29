/*
 * robot_control.h
 *
 *  Created on: Sep 28, 2019
 *      Author: yinon
 */

#ifndef ROBOT_CONTROL_H_
#define ROBOT_CONTROL_H_

#include "opencv2/core.hpp"

#include "serial_comm.h"
#include "pid.h"

class RobotController
{
public:
	RobotController();

	// control the robot as a function of perpendicular distance (meters)
	// and the object's centroid in frame (pixels, relative to frame center)
	void control(double perp_distance, int centroid_x, int centroid_y);

	// control velocity and turn rate directly
	void control(int vel, int turn_rate);
private:
	PIDController velocity_pid;
	PIDController turn_rate_pid;

	ArduinoSerialComm serial_comm;
};

#endif /* ROBOT_CONTROL_H_ */
