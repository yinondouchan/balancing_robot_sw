import sys, pygame
from time import sleep

from teleop import TeleopSerialInterface

pygame.init()

serial_interface = TeleopSerialInterface()

size = width, height = 640, 480
black = 0, 0, 0

screen = pygame.display.set_mode(size)

ball = pygame.image.load("intro_ball.gif")
ballrect = ball.get_rect()

# draw initial screen
screen.fill(black)

joystick_home_pos = [int(width/2), int(height/2)]
joystick_pos = [joystick_home_pos[0], joystick_home_pos[1]]

joystick_xmin = joystick_home_pos[0] - 150
joystick_ymin = joystick_pos[1] - 150
joystick_xmax = joystick_home_pos[0] + 150
joystick_ymax = joystick_pos[1] + 150


def render(pos):
    screen.fill(black)
    pygame.draw.circle(screen, (0, 255, 0), pos, 30)
    pygame.draw.rect(screen, (255, 255, 255), (joystick_xmin, joystick_ymin,
                                               joystick_xmax - joystick_xmin, joystick_ymax - joystick_ymin), 1)


def serial_write_vel_and_turn_rate(pos):
    turn_rate = 512 * (pos[0] - joystick_home_pos[0]) / (joystick_xmax - joystick_home_pos[0])
    velocity = 512 * (pos[1] - joystick_home_pos[1]) / (joystick_ymax - joystick_home_pos[1])

    turn_rate = int(turn_rate)
    velocity = int(velocity)


render(joystick_home_pos)

pygame.display.flip()

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

        if event.type == pygame.MOUSEMOTION:
            left_pressed, _, right_pressed = event.buttons
            #print((event.pos, event.rel, event.buttons))

            if left_pressed:
                joystick_pos[0] += event.rel[0]
                joystick_pos[1] += event.rel[1]

                # clip joystick position to boundaries
                joystick_pos[0] = min(joystick_pos[0], joystick_xmax)
                joystick_pos[0] = max(joystick_pos[0], joystick_xmin)
                joystick_pos[1] = min(joystick_pos[1], joystick_ymax)
                joystick_pos[1] = max(joystick_pos[1], joystick_ymin)

                serial_write_vel_and_turn_rate(joystick_pos)

                render(joystick_pos)

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            joystick_pos = [joystick_home_pos[0], joystick_home_pos[1]]
            serial_write_vel_and_turn_rate(joystick_pos)
            render(joystick_pos)

        pygame.display.flip()
