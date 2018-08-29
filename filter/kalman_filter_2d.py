# xy pos/vel
from filter.kalman_filter import KalmanFilter
import util.config

class KalmanFilter2D(KalmanFilter):
    def __init__(self, initial_state):
        KalmanFilter.__init__(self, 4, 2)

        # State transition matrix (A)
        # F_k
        # Control transition matrix (B)
        # B_k
        # Observation matrix (C)
        # H_k

        # Covariance of process noise (how wrong A is)
        # Q_k
        # Covariance of observation noise (how wrong z_k is)
        # R_k
        pass