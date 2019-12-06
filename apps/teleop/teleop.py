import serial

from getkey import getkey, keys
from time import sleep

# velocity and turn rate are both between 0 and 1024 where 512 is the zero point

# centered velocity and turn rate
velocity = 0
turn_rate = 0

velocity_sensitivity = 20
turn_rate_sensitivity = 75

# open serial port
ser = serial.Serial('/dev/ttyUSB0')

while True:
    k = getkey()

    # move
    if k == keys.UP:
        velocity += velocity_sensitivity
    if k == keys.DOWN:
        velocity -= velocity_sensitivity
    if k == keys.LEFT:
        turn_rate -= turn_rate_sensitivity
    if k == keys.RIGHT:
        turn_rate += turn_rate_sensitivity
    if k == keys.SPACE:
        velocity = 0
        turn_rate = 0

    if k == keys.E:
        turn_rate = 0

    # increase/decrease velocity scale
    if k == keys.Q:
        ser.write(bytearray("B1  ", 'ascii'))
    if k == keys.A:
        ser.write(bytearray("B2  ", 'ascii'))
    

    # get up/fall
    if k == keys.ENTER:
        ser.write(bytearray("B3  ", 'ascii'))
        continue

    # stop teleop (also resets Arduino, thus causing the robot to fall)
    if k == keys.ESC:
        break

    # clip velocity and turn rate
    velocity = max(-512, min(velocity, 512))
    turn_rate = max(-512, min(turn_rate, 512))

    command_str = "S%04d  D%04d  " % (velocity + 512, turn_rate + 512)
    ser.write(bytearray(command_str, 'ascii'))

ser.close()
