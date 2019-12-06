#!/usr/bin/env python

import socket

from teleop import TeleopSerialInterface

serial_interface = TeleopSerialInterface()
serial_interface.open()



TCP_IP = '127.0.0.1'
TCP_PORT = 5005
MAX_MESSAGE_SIZE = 14
BUFFER_SIZE = MAX_MESSAGE_SIZE

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()
print('Connection address:', addr)
while 1:
    data = conn.recv(BUFFER_SIZE)
    print("received data:", data)
    serial_interface.write_to_serial(data)

serial_interface.close()
conn.close()