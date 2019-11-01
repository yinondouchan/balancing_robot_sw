/*
 * arduino_serial_comm.h
 *
 *  Created on: Sep 28, 2019
 *      Author: yinon
 */

#ifndef SERIAL_COMM_H_
#define SERIAL_COMM_H_

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

/*
 * The serial communication interface with the robot's Arduino controller
 */
class ArduinoSerialComm
{
public:
    ArduinoSerialComm();
    void write_serial(const string buf);
    void write_velocity(int vel, int ang_vel);
private:
    int fd;
};


#endif /* SERIAL_COMM_H_ */
