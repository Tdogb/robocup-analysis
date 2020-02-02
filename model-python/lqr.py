import numpy as np
import scipy
import scipy.signal

# A = np.matrix([[-3.40636316, 0, 0],
#                 [0, -3.40636316, 0],
#                 [0, 0, -11.8314673]])
max_timesteps = 500
timestep = 1

fps = 60
dt = 1.0 / fps

P = []

U = []

# A_c = np.matrix([[-3.40636316, 0, 0],
#                 [0, -3.40636316, 0],
#                 [0, 0, -11.8314673]])

# B_c = np.matrix([[-0.34751847, -0.34751847, 0.34751847, 0.34751847],
#                 [0.34751847, -0.34751847, -0.34751847, 0.34751847],
#                 [10.0413389,  10.0413389,  10.0413389, 10.0413389]])

A_c = np.asmatrix([[-3.40636316e+00,  4.16333634e-17, -2.64649413e-14],
                    [6.22967685e-15, -3.40636316e+00,  4.61530416e-14],
                    [-1.67164675e-14,  3.89006384e-15, -1.18314673e+01]])

B_c = np.asmatrix([[-0.86559115,  0.17055422, -0.17055422,  0.86559115],
                    [-0.7721252,   0.7721252,  -1.46716214,  1.46716214],
                    [16.58071345,  3.50196438, 16.58071345,  3.50196438]])

#Discretize A and B
A = np.exp(A_c * dt)

B = np.linalg.inv(A_c) * (A - np.identity(3)) * B_c

Q_param = 1000 #1 / (10*10)
R_param = 10 #1 / (24*24)

Q = np.diag(np.array([Q_param,Q_param,Q_param]))
R = np.diag(np.array([R_param,R_param,R_param,R_param]))

convergedP = np.zeros((3,3))

def init():
    recursivePt(Q) #OR Q

def recursivePt(P_t_1):
    global timestep, convergedP
    if(timestep <= max_timesteps):
        # print(equationPt(P_t_1))
        result = equationPt(P_t_1)
        P.insert(max_timesteps - (timestep - 1), result)
        # calculateU(result, timestep - 1)
        timestep += 1
        if (result-P_t_1).all():
            print("CONVERGED")
            convergedP = result
            return 0 #<-----------------!!!!!!!!!!!!
        return recursivePt(result)
    else:
        result = equationPt(P_t_1)
        P.insert(0, result)
        # calculateU(result, timestep - 1)
        return result

def equationPt(P_t_1):
    return Q + A.T * P_t_1 * A - A.T * P_t_1 * B * np.linalg.inv(R + B.T * P_t_1 * B) * B.T * P_t_1 * A

def calculateU(P_t_1, xt, t):
    # v = 3
    # xt = v * np.asmatrix([np.cos(v * t), -np.sin(v * t / 3) / 3, np.cos(t)]).T
    ut = -1 * np.linalg.inv(R + B.T * P_t_1 * B) * B.T * P_t_1 * A * xt
    return ut
    # U.insert(t, ut)

def controlLQR(x, t):
    return calculateU(convergedP, x, int(t)) #P[int(t+1)]

if __name__ == "__main__":
    init()