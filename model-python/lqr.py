import numpy as np
import scipy
import scipy.signal

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

A_c = np.matrix([[-4.53100603e+00, -4.90302243e-14, -6.98555797e-14],
                 [ 5.74831579e-13, -2.23582524e+00,  3.14349369e-02],
                 [ 2.14046085e-13,  8.70171263e+00, -1.18272693e+01]])

B_c = np.matrix([[-0.39668424, -0.39668424,  0.39668424,  0.39668424],
                 [ 0.25590858, -0.3092661,  -0.3092661,   0.25590858],
                 [10.075514,   10.00003824, 10.00003824, 10.075514  ]])

#Discretize A and B
A = np.asmatrix(scipy.linalg.expm(A_c * dt))

B = np.linalg.inv(A_c) * (A - np.identity(3)) * B_c

A_new = np.zeros((6,6))
A_new[3:, 3:] = A
A_new[0:3, 3:] = dt * np.identity(3)
A_new[:3,:3] = np.identity(3) # currentX - dt*vel
print(A_new)

B_new = np.asmatrix(np.zeros((6,4)))
B_new[3:,:] = B

print(B_new)

A = A_new
B = B_new

Q_param = 500 #1 / (10*10)
R_param = 10 #1 / (24*24)

Q = np.asmatrix(np.diag(np.array([Q_param*10,Q_param*10,Q_param*10,Q_param,Q_param,Q_param])))
R = np.asmatrix(np.diag(np.array([R_param,R_param,R_param,R_param])))#np.diag(np.array([R_param,R_param,R_param,R_param,R_param,R_param]))

convergedP = np.zeros((3,3))
controlMatrix = np.zeros((4,6))

def init():
    np.set_printoptions(linewidth=200)
    recursivePt(Q) #OR Q

def recursivePt(P_t_1):
    global timestep, convergedP, controlMatrix
    if(timestep <= max_timesteps):
        result = equationPt(P_t_1)
        P.insert(max_timesteps - (timestep - 1), result)
        timestep += 1
        if np.all((result-P_t_1) < 1e-8):
            print("CONVERGED")
            convergedP = result
            controlMatrix = np.linalg.inv(R + B.T * convergedP * B) * B.T * convergedP * A
            print(controlMatrix)
            return 0 #<-----------------!!!!!!!!!!!!
        return recursivePt(result)
    else:
        result = equationPt(P_t_1)
        P.insert(0, result)
        return result

def equationPt(P_t_1):
    return Q + A.T @ P_t_1 @ A - A.T @ P_t_1 @ B @ np.linalg.inv(R + B.T * P_t_1 @ B) @ B.T @ P_t_1 @ A

def controlLQR(x):
    return controlMatrix @ x

if __name__ == "__main__":
    init()