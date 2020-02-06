import pygame
import model
import numpy as np
from controller import Controller
import lqr
import vis
import motor
import robot
from util import rotation_matrix
import random
import csv

robot_data = [[]]
def main():
    lqr.init()
    clock = pygame.time.Clock()

    pos = np.asmatrix([0., 1., 0.]).T
    vel = np.asmatrix([3., 0., 3.]).T

    visualizer = vis.Visualizer()

    fps = 60
    dt = 1.0 / fps
    t = 0.0
    i = 0

    maxon_motor = motor.Motor(
        resistance=1.03,
        torque_constant=0.0335,
        speed_constant=0.0335,
        rotor_inertia=135*1e-3*(1e-2**2))
    gearbox = motor.Gearbox(gear_ratio=20.0 / 60.0, gear_inertia=0)
    gear_motor = motor.GearMotor(maxon_motor, gearbox)
    our_robot = robot.Robot(
        robot_mass=6.5,
        robot_radius=0.085,
        gear_motor=gear_motor,
        robot_inertia=6.5*0.085*0.085*0.5,
        wheel_radius=0.029,
        wheel_inertia=2.4e-5,
        wheel_angles=np.deg2rad([45, 135, -135, -45]))

    controller = Controller(dt, our_robot)

    while not visualizer.close:
        visualizer.update_events()

        v = 3

        rx = np.asmatrix([np.sin(v * t), np.cos(v * t / 3), v * np.sin(t)]).T
        rv = v * np.asmatrix([np.cos(v * t), -np.sin(v * t / 3) / 3, np.cos(t)]).T
        ra = v ** 2 * np.asmatrix([-np.sin(v * t), -np.cos(v * t / 3) / 9, -np.sin(t) / v]).T

        # u = lqr.controlLQR(rv-vel, t * 60)
        # u = controller.feedforward_control(pos, vel, rx, rv, ra)
        u = controller.control(pos, vel, rx, rv, ra)

        vdot = our_robot.forward_dynamics_world(pos, vel, u)

        vel_b = np.linalg.inv(rotation_matrix(pos[2,0])) * vel
        vdot_b = our_robot.forward_dynamics_body(vel_b, u)

        # print(u - our_robot.inverse_dynamics_body(vel_b, vdot_b))

        robot.sysID(u[0,0], u[1,0], u[2,0], u[3,0], vel_b[0,0], vel_b[1,0], vel_b[2,0], vdot_b[0,0], vdot_b[1,0], vdot_b[2,0])
        
        visualizer.draw(pos, rx)
        pos += dt * vel + 0.5 * vdot * dt ** 2
        vel += dt * vdot
        # robot_data_line = [vdot_b[0,0], vdot_b[1,0], vdot_b[2,0], vel_b[0,0], vel_b[1,0], vel_b[2,0], u[0,0], u[1,0], u[2,0], u[3,0]]
        # robot_data.append(robot_data_line)

        clock.tick(60)
        t += dt
        i += 1
        if t > 20:
            print("t = 20")
            t = 0.0
            i = 0
            pos = np.asmatrix([0, 1, 3.]).T
            vel = np.asmatrix([1, 0, 1.]).T
            controller.reset()
    robot.main()
    # writeCSV()

def writeCSV():
    with open('robot_data.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
        for line in robot_data:
            writer.writerow(line)

if __name__ == '__main__':
    main()