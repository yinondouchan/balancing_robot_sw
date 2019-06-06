#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

class ArduinoSerialComm
{
public:
    ArduinoSerialComm();
    void write_serial(const string buf);
    void write_velocity(int vel, int ang_vel);
private:
    int fd;
};

