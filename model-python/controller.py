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
        self.ff_coefs = np.matrix([[-3.47518469e-01],
                        [-3.47518469e-01],
                        [ 3.47518469e-01],
                        [ 3.47518469e-01],
                        [-3.40636316e+00],
                        [-2.48412402e-15],
                        [ 1.63454320e-15],
                        [-3.33066907e-16],
                        [ 9.49457230e-01],
                        [-3.32719963e-15],
                        [ 3.47518469e-01],
                        [-3.47518469e-01],
                        [-3.47518469e-01],
                        [ 3.47518469e-01],
                        [-3.11852353e-14],
                        [-3.40636316e+00],
                        [ 7.73427692e-14],
                        [-9.49457230e-01],
                        [-9.41468167e-16],
                        [-1.16616943e-14],
                        [ 1.00413389e+01],
                        [ 1.00413389e+01],
                        [ 1.00413389e+01],
                        [ 1.00413389e+01],
                        [ 3.21517864e-15],
                        [ 3.76372433e-16],
                        [-1.18314673e+01],
                        [-1.82042895e-15],
                        [ 3.60464342e-16],
                        [ 1.88173939e-14]])

    def reset(self):
        self.integral = np.asmatrix(np.zeros((3, 1)))

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

    #Don't need x for pure ff

    # ss_A = np.matrix([[-3.40636316, 0, 0],
    #                   [0, -3.40636316, 0],
    #                   [0, 0, -11.8314673]])

    # ss_B = np.matrix([[-0.34751847, -0.34751847, 0.34751847, 0.34751847],
    #                   [0.34751847, -0.34751847, -0.34751847, 0.34751847],
    #                   [10.0413389,  10.0413389,  10.0413389, 10.0413389]])

    def feedforward_control(self, current_x, vel, rx, rv, ra):
        A = np.matrix([[-3.40636316, 0, 0, 0, 0.949457230],
                       [0, -3.40636316, 0, -0.949457230, 0],
                       [0, 0, -11.8314673, 0, 0]])

        B = np.matrix([[-0.34751847, -0.34751847, 0.34751847, 0.34751847],
                       [0.34751847, -0.34751847, -0.34751847, 0.34751847],
                       [10.0413389,  10.0413389,  10.0413389, 10.0413389]])
                       
        rv_body = np.linalg.inv(util.rotation_matrix(current_x[2,0])) * rv
        xDot = np.linalg.inv(util.rotation_matrix(current_x[2,0])) * ra

        x = np.matrix([[rv_body[0,0]],
                       [rv_body[1,0]],
                       [rv_body[2,0]],
                       [rv_body[0,0]*rv_body[2,0]],
                       [rv_body[1,0]*rv_body[2,0]]])
        return np.linalg.pinv(B)*(xDot - A*x)
        '''
        # if x[2,0] > math.pi:
        #     return np.matrix([0,0,0,0]).T
        # print(np.linalg.inv(util.rotation_matrix(math.pi/4))*np.matrix([[1],[1],[0]]))
        x_body = np.linalg.inv(util.rotation_matrix(x[2,0])) * x
        vel_body = np.linalg.inv(util.rotation_matrix(x[2,0])) * vel
        rv_body = np.linalg.inv(util.rotation_matrix(x[2,0])) * rv
        ra_body = np.linalg.inv(util.rotation_matrix(x[2,0])) * ra

        vels_world = np.matrix([rv[0,0],
                                rv[1,0],
                                rv[2,0],
                                rv[0,0]*rv[2,0],
                                rv[1,0]*rv[2,0]]).T

        vels = np.matrix([rv_body[0,0],
                          rv_body[1,0],
                          rv_body[2,0],
                          rv_body[0,0]*rv_body[2,0],
                          rv_body[1,0]*rv_body[2,0]]).T

        # print(np.vstack((np.vstack((self.ff_coefs.T[:,4:9],self.ff_coefs.T[:,14:19])),self.ff_coefs.T[:,24:29])))
        # print(vels)
        VCMat = np.vstack((np.vstack((self.ff_coefs.T[:,4:9], self.ff_coefs.T[:,14:19])), self.ff_coefs.T[:,24:29]))
        constantTermMatrix = np.matrix([[self.ff_coefs.T[0,9]], [self.ff_coefs.T[0,19]], [self.ff_coefs.T[0,29]]])
        MCMat = np.vstack((np.vstack((self.ff_coefs.T[:,0:4], self.ff_coefs.T[:,10:14])), self.ff_coefs.T[:,20:24]))
        print("-----------")
        print(VCMat)
        print(constantTermMatrix)
        print(MCMat)
        print(np.linalg.pinv(MCMat))
        print(x[2,0])
        solvedParamX = np.matmul(np.linalg.pinv(MCMat), (ra_body - np.matmul(VCMat, vels) - constantTermMatrix))
        
        a = np.matmul(np.linalg.pinv(MCMat), (ra - np.matmul(VCMat, vels_world) - constantTermMatrix))
        b = self.model.inverse_dynamics_body(rv, ra)
        # print(ra)
        # print(a)
        # print(b)
        # print("-------------")
        # print(solvedParamX)
        # print(self.model.inverse_dynamics_world(x, vel, ra))
        # print(solvedParamX/self.model.inverse_dynamics_world(x, vel, ra))
        # print("-------------")

        # print("-------")
        # np.vstack(self.ff_coefs.T[:,0:5],self.ff_coefs.T[:,0:5])
        # print(np.matmul(solvedParamX,np.linalg.pinv(self.ff_coefs[0:4,:])))
        return solvedParamX#np.matmul(solvedParamX,np.linalg.pinv(self.ff_coefs[0:4,:])).T
        '''