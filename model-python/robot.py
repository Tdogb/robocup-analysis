"""Robocup robot modelling."""

import numpy as np
import math
import matplotlib.pyplot as plt
import motor
import util


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

Vsys = np.matrix([[0.0],[0.0],[0.0]]) #x, y, theta
currentTimestep = 0
numTimesteps = 70
dt = 0.01
# volts = 24

# A = BC

# AMeasured = np.zeros((numTimesteps*3,1))
Ax = np.zeros((numTimesteps,1))
Ay = np.zeros((numTimesteps,1))
At = np.zeros((numTimesteps,1))

# Variables are Vx, Vy, Vz, U1, U2, U3, U4, Vx*Vtheta, Vy*Vtheta

Bx = np.zeros((numTimesteps,10))
By = np.zeros((numTimesteps,10))
Bt = np.zeros((numTimesteps,10))

Cx = np.zeros((10,1))
Cy = np.zeros((10,1))
Ct = np.zeros((10,1))

Vmeasured = np.zeros((numTimesteps*3, 1))

def sysId(robot, voltages):
    global Vsys
    global currentTimestep
    global dt
    #print(voltages)
    Asys = robot.forward_dynamics_body(Vsys, voltages.T)
    Ax[currentTimestep,0] = Asys[0]
    Ay[currentTimestep,0] = Asys[1]
    At[currentTimestep,0] = Asys[2]
    Vsys += Asys * dt
    #Voltages = np.matrix([[24,24,24,24]])
    
    BRowTempLin = np.concatenate((Vsys.T, voltages),axis=1)
    VsysT = Vsys.T
    BRowTempSq = np.matrix([[VsysT[0,0]*VsysT[0,2],VsysT[0,1]*VsysT[0,2]]])
    BRowTemp = np.concatenate((BRowTempLin, BRowTempSq), axis=1)
    BRow = np.concatenate((BRowTemp, np.matrix([1])),axis=1)
    Bx[currentTimestep,:] = BRow
    By[currentTimestep,:] = BRow
    Bt[currentTimestep,:] = BRow
    # print(currentTimestep)
    currentTimestep += 1

def test_sysid(robot, vels, Cx, Cy, Ct):
    volts = np.matrix([[-24,24,-24,24]])
    # print(vels)
    # print("  ")
    # print(volts.T)
    # print(vels.shape)
    row = np.concatenate((vels.T, volts), axis=1)
    secondArea = np.matrix([vels[0,0]*vels[2,0], vels[1,0]*vels[2,0], 1])
    row2 = np.concatenate((row, secondArea), axis=1)
    # print(row2)
    # print(Cx)
    print(row2*Cx)
    actualParams = robot.forward_dynamics_body(vels, volts.T)
    print(actualParams)
    # print(Cx)

    #CxInv = np.linalg.pinv(Cx)
    # print(Cx)
    # print("blank")
    # print(2*accel[0] * -Cx.T)
    # vels = np.matrix([[0],[0],[0]])
    # actualParams = robot.inverse_dynamics_body(vels, accel)
    # print(actualParams)

def main():
    global Ax
    global Ay
    global At
    global Bx
    global By
    global Bt
    global Cx
    global Cy
    global Ct
    global numTimesteps
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
    voltage = np.asmatrix([12, 24, 24, 24]).T
    accel = np.asmatrix([1, 2, 3]).T
    pose = np.asmatrix([1, 2, 1]).T
    vc = 0
    ac = 0
    maxVel = 10
    maxVelMat = np.matrix([[maxVel],[0],[0]])
    climbTimesteps = 30
    cruiseTimesteps = 10
    numTimesteps = climbTimesteps*2 + cruiseTimesteps
    climbAccleration = maxVel/climbTimesteps

    for i in np.arange(0,maxVel,maxVel/climbTimesteps):
        a = np.matrix([[climbAccleration],[0],[0]])
        v = np.matrix([[i],[i],[0]])
        voltages = robot.inverse_dynamics_body(v,a).T
        sysId(robot,voltages)
    for _ in range(0, 10): #Timesteps to be in cruise for
        zeroAccel = np.matrix([[0],[0],[0]])
        voltages = robot.inverse_dynamics_body(maxVelMat,zeroAccel).T
        sysId(robot, voltages)
    for i in np.arange(maxVel,0,-maxVel/climbTimesteps):
        a = np.matrix([[-climbAccleration],[0],[0]])
        v = np.matrix([[i],[0],[i]])
        voltages = robot.inverse_dynamics_body(v,a).T
        sysId(robot,voltages)

    # for t in range(0,100):
        # a = np.matrix([math.])

    BxInv = np.linalg.pinv(Bx)
    ByInv = np.linalg.pinv(By)
    BtInv = np.linalg.pinv(Bt)
    # print(BxInv)
    # print(Ax)
    # print("done")
    Cx = np.matmul(BxInv,Ax)
    Cy = np.matmul(ByInv,Ay)
    Ct = np.matmul(BtInv,At)
    vels = np.matrix([[4],[0.5],[0]])
    test_sysid(robot, vels, Cx, Cy, Ct)
    # print(Ax)
    # print(Bx)
    # print(Cx)

if __name__ == '__main__':
    main()
