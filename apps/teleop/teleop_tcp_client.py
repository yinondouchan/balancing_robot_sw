#!/usr/bin/env python

import socket
from teleop import TeleopSerialInterface


class TeleopTCPClient:

    MAX_MESSAGE_SIZE = 14
    BUFFER_SIZE = MAX_MESSAGE_SIZE

    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._socket = None

    def connect(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._ip, self._port))

    def set_velocity_and_turn_rate(self, velocity, turn_rate):
        """ set velocity and turn rate. Both in range of -512 to 512 """
        command_str = "S%04d  D%04d  " % (velocity + 512, turn_rate + 512)
        self._socket.send(bytearray(command_str, 'ascii'))

    def increase_velocity_scale(self):
        self._socket.send(bytearray("B1  ", 'ascii'))

    def decrease_velocity_scale(self):
        self._socket.send(bytearray("B2  ", 'ascii'))

    def close(self):
        self._socket.close()
