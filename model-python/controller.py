import numpy as np
import util
import math

class Controller:
    def __init__(self, dt, model):
        self.integral = np.asmatrix(np.zeros((3, 1)))
        self.dt = dt
        self.Kp, self.Ki, self.Kd = (50, 0, 5)
        self.model = model
        self.Mp = np.diag([50, 50, 100])
        self.Mi = 0 * np.diag([50, 50, 100])
        self.Md = np.diag([15, 15, 60])

    def reset(self):
        self.integral = np.asmatrix(np.zeros((3, 1)))

    def convertToBody(self, x, vec):
        return np.linalg.inv(util.rotation_matrix(x[2,0])) * vec

    def control(self, x, v, rx, rv, ra):
        gRb = util.rotation_matrix(x[2, 0])

        gRp = util.rotation_matrix(rx[2, 0])
        error_path_relative = gRp * (rx - x)

        self.integral += self.dt * error_path_relative

        G = self.model.geom

        GTinv = np.linalg.pinv(G.T * 0.029)
        error = gRb.T * (rx - x)
        error[2, 0] *= 0.1

        derivative = gRb.T * (rv - v)
        derivative[2, 0] *= 0.01
        goal_acceleration = ra + self.Mp * (rx - x) + self.Md * (rv - v) + self.Mi * gRp.T * self.integral
        return self.model.inverse_dynamics_world(x, rv, goal_acceleration)
        uff = self.model.inverse_dynamics_world(x, rv, ra)
        up = self.Kp * GTinv * error
        ui = self.Ki * GTinv * np.diag([1, 1, 0]) * gRb * gRp.T * self.integral
        ud = self.Kd * GTinv * derivative
        return uff + up + ui + ud

    def feedforward_control(self, current_x, vel, rx, rv, ra):
        A = np.asmatrix([[-3.40636316e+00,  1.72084569e-15, -2.22044605e-16,  3.33066907e-16, 9.49457230e-10],
                         [ 1.06332511e-15, -3.40636316e+00, -4.89821158e-15, -9.49457230e-10, 5.97084703e-15],
                         [ 8.05604854e-15, -3.39407337e-16, -1.18314673e+01,  1.29496126e-16, -4.45369336e-15]])

        B = np.asmatrix([[-0.34751847, -0.34751847,  0.34751847,  0.34751847],
                         [ 0.34751847, -0.34751847, -0.34751847,  0.34751847],
                         [10.04133891, 10.04133891, 10.04133891, 10.04133891]])

        constantMatrix = np.matrix([[-2.88657986e-15],
                                    [-2.66501017e-14],
                                    [ 1.58923465e-14]])

        rv_body = np.linalg.inv(util.rotation_matrix(current_x[2,0])) * rv
        vel_body = np.linalg.inv(util.rotation_matrix(current_x[2,0])) * vel
        xDot = np.linalg.inv(util.rotation_matrix(current_x[2,0])) * ra

        x = np.matrix([[rv_body[0,0]],
                       [rv_body[1,0]],
                       [rv_body[2,0]],
                       [rv_body[0,0]*rv_body[2,0]],
                       [rv_body[1,0]*rv_body[2,0]]])
        feedforward = np.linalg.pinv(B)*(xDot - A*x - constantMatrix)
        return feedforward

    def path_PID_control(self, current_x, vel, rx, rv):
        global convertToBody
        error = convertToBody(current_x, rx) - convertToBody(current_x, current_x)
        p = 1 * error
        return p
