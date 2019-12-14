#!/usr/bin/env python

import socket
import sys

from teleop import TeleopSerialInterface

serial_interface = TeleopSerialInterface()
serial_interface.open()

if (len(sys.argv) < 3):
    TCP_IP = '10.42.0.1'
    TCP_PORT = 5005
else:
    TCP_IP = socket.gethostbyname(sys.argv[1])
    TCP_PORT = int(sys.argv[2])

MAX_MESSAGE_SIZE = 14
BUFFER_SIZE = MAX_MESSAGE_SIZE

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

while True:
    conn, addr = s.accept()
    print('Connection address:', addr)
    while True:
        data = conn.recv(BUFFER_SIZE)
        if not data:
            break
        serial_interface.write_to_serial(data)

serial_interface.close()
conn.close()
