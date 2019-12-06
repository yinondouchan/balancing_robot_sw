import serial

class TeleopSerialInterface:
    """
    Serial interface for teleoperation of the robot
    """

    def __init__(self, serial_device_name='/dev/ttyUSB0'):
        self._serial_device_name = serial_device_name
        self._serial_connection = None

    def open(self):
        self._serial_connection = serial.Serial(self._serial_device_name)

    def set_velocity_and_turn_rate(self, velocity, turn_rate):
        """ set velocity and turn rate. Both in range of -512 to 512 """
        command_str = "S%04d  D%04d  " % (velocity + 512, turn_rate + 512)
        self._serial_connection.write(bytearray(command_str, 'ascii'))

    def increase_velocity_scale(self):
        self._serial_connection.write(bytearray("B1  ", 'ascii'))

    def decrease_velocity_scale(self):
        self._serial_connection.write(bytearray("B2  ", 'ascii'))

    def close(self):
        self._serial_connection.close()
