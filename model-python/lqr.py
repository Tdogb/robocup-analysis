import numpy as np
import scipy
import scipy.signal

max_timesteps = 500
timestep = 1

fps = 60
dt = 1.0 / fps

P = []

U = []

A_c = np.matrix([[-3.40636316, 0, 0],
                [0, -3.40636316, 0],
                [0, 0, -11.8314673]])

B_c = np.matrix([[-0.34751847, -0.34751847, 0.34751847, 0.34751847],
                [0.34751847, -0.34751847, -0.34751847, 0.34751847],
                [10.0413389,  10.0413389,  10.0413389, 10.0413389]])

#Discretize A and B
A = np.asmatrix(scipy.linalg.expm(A_c * dt))

B = np.linalg.inv(A_c) * (A - np.identity(3)) * B_c

# print("Discrete A:")
# print(A)
A_new = np.zeros((6,6))
A_new[3:, 3:] = A
A_new[0:3, 3:] = -dt * np.identity(3)
A_new[:3,:3] = np.identity(3) # currentX - dt*vel
print(A_new)

B_new = np.zeros((6,4))
B_new[3:,:] = B
# B_new = np.zeros((6,4))
# B_new[3:,:] = B
print(B_new)

A = A_new
B = B_new

Q_param = 5000 #1 / (10*10)
R_param = 1 #1 / (24*24)

Q = np.diag(np.array([0,0,0,Q_param,Q_param,Q_param]))
R = np.diag(np.array([R_param,R_param,R_param,R_param]))#np.diag(np.array([R_param,R_param,R_param,R_param,R_param,R_param]))

convergedP = np.zeros((3,3))
controlMatrix = np.zeros((4,6))

def init():
    global controlMatrix
    np.set_printoptions(linewidth=200)
    controlMatrix = scipy.linalg.solve_discrete_are(A,B,Q,R)
    # recursivePt(Q) #OR Q

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
            print(convergedP)
            return 0 #<-----------------!!!!!!!!!!!!
        return recursivePt(result)
    else:
        result = equationPt(P_t_1)
        P.insert(0, result)
        return result

def equationPt(P_t_1):
    # print(B.T * P_t_1)
    # print(scipy.linalg.solve_discrete_are(A,B,Q,R))
    print("------------------------------------------------")
    print(Q + A.T * P_t_1 * A - A.T * P_t_1)
    print(B)
    print("=---------------------pass---------------------=")
    # print(P_t_1)
    # print(Q + A.T * P_t_1 * A - A.T * P_t_1 * B * np.linalg.inv(R + B.T * P_t_1 * B) * B.T * P_t_1 * A)
    return Q + A.T * P_t_1 * A - A.T * P_t_1 * B * np.linalg.inv(R + B.T * P_t_1 * B) * B.T * P_t_1 * A

def controlLQR(x):
    print(controlMatrix * x)
    return controlMatrix * x

if __name__ == "__main__":
    init()
