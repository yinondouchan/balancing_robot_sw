#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "arduino_serial_comm.h"

#define VEL_ZERO_POINT 512
#define ANG_VEL_ZERO_POINT 512

using namespace std;

ArduinoSerialComm::ArduinoSerialComm()
{
    struct termios options;
    memset (&options, 0, sizeof options);
    int flags;

    
    fd = open("/dev/ttyUSB0", O_WRONLY | O_NOCTTY);
    if (fd < 0)
    {
        cout << "Unable to open ofstream to Arduino" << endl << flush;
    }
    else
    {
        cout << "ofstream to Arduino opened successfully" << endl << flush;
        if(tcgetattr(fd, &options) != 0)
        {
            cout<<"ERR: could not get terminal options" << endl << flush;
        }
        else
        {
            /*options.c_cflag &= ~CRTSCTS;    
            options.c_cflag |= (CLOCAL | CREAD);                   
            options.c_iflag |= (IGNPAR | IGNCR);                  
            options.c_iflag &= ~(IXON | IXOFF | IXANY);          
            options.c_oflag &= ~OPOST;
            options.c_iflag &= ~INPCK;         
            options.c_iflag &= ~(ICRNL|IGNCR);  
            options.c_iflag |= INPCK;*/
            

            options.c_cflag &= ~CSIZE;            
            options.c_cflag |= CS8;              
            options.c_cflag &= ~PARENB;
            options.c_cflag &= ~CSTOPB;
            
            /* no hardware flow control */
			options.c_cflag &= ~CRTSCTS;
			/* enable receiver, ignore status lines */
			options.c_cflag |= CREAD | CLOCAL;
			/* disable input/output flow control, disable restart chars */
			options.c_iflag &= ~(IXON | IXOFF | IXANY);
			/* disable canonical input, disable echo,
			disable visually erase chars,
			disable terminal-generated signals */
			options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
			/* disable output processing */
			options.c_oflag &= ~OPOST;
      
            options.c_cc[VTIME] = 0.001;  //  1s=10   0.1s=1 *
            options.c_cc[VMIN] = 0;

        	cfsetispeed(&options, (speed_t)B9600);
            cfsetospeed(&options, (speed_t)B9600);

            tcflush(fd, TCIFLUSH);
            if(tcsetattr(fd, TCSANOW, &options) < 0)
            {
                cout<<"\n\nERR: could not set new terminal options\n\n";
            }
            
            //ioctl(fd,TIOCMBIS, TIOCM_RTS); //Set RTS pin
		    //ioctl(fd,TIOCMBIS, TIOCM_DTR); //Set DTR pin
		    //ioctl(fd, TIOCMGET, &flags); //Set DTR pin
		    //flags |= TIOCM_DTR;
		    //ioctl(fd, TIOCMSET, &flags);
        }

        sleep(3);
    }
}

void ArduinoSerialComm::write_velocity(int vel, int ang_vel)
{
	// clamp velocity
	vel = vel < -VEL_ZERO_POINT ? -VEL_ZERO_POINT : vel;
	vel = vel > (VEL_ZERO_POINT - 1) ? (VEL_ZERO_POINT - 1) : vel;
	
	// clamp angular velocity
	ang_vel = ang_vel < -ANG_VEL_ZERO_POINT ? -ANG_VEL_ZERO_POINT : ang_vel;
	ang_vel = ang_vel > (ANG_VEL_ZERO_POINT - 1) ? (ANG_VEL_ZERO_POINT - 1) : ang_vel;
	
	vel += VEL_ZERO_POINT;
	ang_vel += ANG_VEL_ZERO_POINT;
	
    std::stringstream ss;
    ss << 'S' << setw(4) << setfill('0') << vel << "\r\nD" << setw(4) << setfill('0') << ang_vel << "\r\n";
    string s = ss.str();

    int size = s.size();

    //cout << s << endl << flush;

    int nbytes = write(fd, s.c_str(), size);
    if (nbytes < 0)
    {
        //cout << "Failed to write. errno " << errno << endl << flush; 
    }
}
