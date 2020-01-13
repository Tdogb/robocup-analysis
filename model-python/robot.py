"""Robocup robot modelling."""

import numpy as np
import motor
import util
import random
import math

class Robot:
    """A robocup robot."""

    def __init__(self,
                 robot_mass,
                 robot_radius,
                 gear_motor,
                 robot_inertia,
                 wheel_radius,
                 wheel_inertia,
                 wheel_angles):
        """Create a new robot."""
        # wheel (angular) velocity = geom * body velocity
        # geom.T * wheel torque = body wrench
        self.geom = np.asmatrix(
            [[-np.sin(th), np.cos(th), robot_radius] for th in wheel_angles]
        ) / wheel_radius

        self.gear_motor = gear_motor
        self.robot_mass_mat = np.asmatrix([
            [robot_mass, 0, 0],
            [0, robot_mass, 0],
            [0, 0, robot_inertia],
        ])

        identity = np.asmatrix(np.eye(4))
        wheel_inertia = (wheel_inertia + gear_motor.inertia()) * identity
        wheel_mass_mat = self.geom.T * wheel_inertia * self.geom
        self.total_mass_mat = self.robot_mass_mat + wheel_mass_mat

        self.wheel_friction_mat = np.asmatrix(np.eye(3)) * 0

    def forward_dynamics_body(self, velocity, voltage):
        """
        Run the continuous model to get a robot acceleration twist.

        Calculations are run in the inertial frame that currently coincides
        with the body frame.

        params:
         velocity: body-space velocity vector [x; y; theta]
         voltage: wheel voltage vector [v1; v2; v3; v4]
        """
        wheel_velocity = self.geom * velocity
        torque = self.gear_motor.forward_dynamics(wheel_velocity, voltage)
        effort_body = self.geom.T * torque

        coriolis = np.asmatrix([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]) * velocity[2, 0]

        drag_matrix = self.wheel_friction_mat + coriolis * self.robot_mass_mat

        return np.linalg.inv(self.total_mass_mat) * (
            effort_body - drag_matrix * velocity)

    def inverse_dynamics_body(self, velocity, acceleration):
        """
        Find the voltage corresponding to a particular acceleration twist.

        params:
         velocity: body-space velocity vector [x; y; theta]
         acceleration: desired body-space acceleration vector [x; y; theta]
        """
        wheel_velocity = self.geom * velocity

        coriolis = np.asmatrix([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]) * velocity[2, 0]

        geom_inv = np.linalg.pinv(self.geom)

        drag_matrix = self.wheel_friction_mat + coriolis * self.robot_mass_mat
        net_torque = geom_inv.T * self.total_mass_mat * acceleration
        drag_torque = geom_inv.T * drag_matrix * geom_inv * wheel_velocity
        required_torque = net_torque + drag_torque

        result = self.gear_motor.inverse_dynamics(
            wheel_velocity, required_torque)
        return result

    def forward_dynamics_world(self, pose, velocity, voltage):
        """
        Perform the forward dynamics calculation in world space.

        params:
         pose: world-space pose vector [x; y; theta]
         velocity: world-space velocity vector [x; y; theta]
         voltage: wheel voltage vector [v1; v2; v3; v4]
        """
        coriolis = np.asmatrix([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]) * velocity[2, 0]

        rotation_matrix = util.rotation_matrix(pose[2, 0])
        velocity_body = rotation_matrix.T * velocity
        acceleration_body = self.forward_dynamics_body(velocity_body, voltage)
        return coriolis * velocity + rotation_matrix * acceleration_body

    def inverse_dynamics_world(self, pose, velocity, acceleration):
        """
        Perform the forward dynamics calculation in world space.

        params:
         pose: world-space pose vector [x; y; theta]
         velocity: world-space velocity vector [x; y; theta]
         voltage: wheel voltage vector [v1; v2; v3; v4]
        """
        coriolis = np.asmatrix([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]) * velocity[2, 0]

        rotation_matrix = util.rotation_matrix(pose[2, 0])
        velocity_body = rotation_matrix.T * velocity
        acceleration_body = rotation_matrix.T * acceleration - \
            coriolis * velocity_body
        return self.inverse_dynamics_body(velocity_body, acceleration_body)

'''
Inputs: Voltage, linear velocity, angular velocity

Ouputs: acceleration

A Matrix: 
vM0     vM1     vM2     vM3       lin vel x       lin vel y       ang vel       angvel*linvelx      angvel*linvely   0       0       0       0         0               0               0        = acceleration x
0       0       0       0         0               0               0         vM0     vM1     vM2     vM3       lin vel x       lin vel y       ang vel  = acceleration y
vM0     vM1     vM2     vM3       lin vel x       lin vel y       ang vel  = acceleration ang

b Matrix: Accelerations

'''
A = np.zeros((3, 30))
b = np.zeros((3, 1))
def sysID(m0Volts, m1Volts, m2Volts, m3Volts, velX, velY, velTh, accelX, accelY, accelTh):
    global A, b
    # print(A)
    # print("-------------------------------------------------")
    A_Block = np.matrix([
        [m0Volts, m1Volts, m2Volts, m3Volts, velX, velY, velTh, velTh*velX, velTh*velY, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m0Volts, m1Volts, m2Volts, m3Volts, velX, velY, velTh, velTh*velX, velTh*velY, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m0Volts, m1Volts, m2Volts, m3Volts, velX, velY, velTh, velTh*velX, velTh*velY, 1]
    ])
    b_block = np.asmatrix([
        [accelX],
        [accelY],
        [accelTh]
    ])
    A = np.concatenate((A,A_Block))
    b = np.concatenate((b,b_block))
    # print(b)


def main():
    global A, b
    maxon_motor = motor.Motor(
        resistance=1.03,
        torque_constant=0.0335,
        speed_constant=0.0335,
        rotor_inertia=135*1e-3*(1e-2**2))
    gearbox = motor.Gearbox(gear_ratio=20.0 / 60.0, gear_inertia=0)
    gear_motor = motor.GearMotor(maxon_motor, gearbox)
    robot = Robot(
        robot_mass=6.5,
        robot_radius=0.085,
        gear_motor=gear_motor,
        robot_inertia=6.5*0.085*0.085*0.5,
        wheel_radius=0.029,
        wheel_inertia=2.4e-5,
        wheel_angles=np.deg2rad([45, 135, -135, -45]))
    velocity = np.asmatrix([-1.0, -2, 3]).T
    voltage = np.asmatrix([24.0, 24, 24, 24]).T
    accel = np.asmatrix([1, 2, 3]).T
    pose = np.asmatrix([1, 2, 1]).T

    for _ in range(0, 100):
        volts = np.asmatrix([[random.uniform(-24,24)],
                             [random.uniform(-24,24)],
                             [random.uniform(-24,24)],
                             [random.uniform(-24,24)]])

        vels = np.asmatrix([[random.uniform(-10, 10)],
                            [random.uniform(-10, 10)],
                            [random.uniform(-10, 10)]])
        accels = robot.forward_dynamics_body(vels, volts)
        # print(accels[0,0])
        sysID(volts[0,0], volts[1,0], volts[2,0], volts[3,0], vels[0,0], vels[1,0], vels[2,0], accels[0,0], accels[1,0], accels[2,0])
    lstsq_solution = np.linalg.lstsq(A,b)

    varsLin = np.vstack((volts,vels))
    varsSq = np.asmatrix([
        [vels[0,0]*vels[2,0]],
        [vels[1,0]*vels[2,0]],
        [1]
    ])
    vars = np.vstack((varsLin, varsSq))
    estimatedAccels = np.array([
        np.matmul(lstsq_solution[0][0:10,0].T,vars),
        np.matmul(lstsq_solution[0][10:20,0].T,vars),
        np.matmul(lstsq_solution[0][20:30,0].T,vars)
    ])
    # print(lstsq_solution[0])
    
    print("--------")
    print(estimatedAccels)
    print(accels)
    

if __name__ == '__main__':
    main()

