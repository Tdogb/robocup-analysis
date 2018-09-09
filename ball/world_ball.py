# Ball in the world space (not single camera space) which consists of one or more merged kalman balls
from ball.kalman_ball import KalmanBall
import util.config
import numpy as np
import math

class WorldBall:
    def __init__(self, kalman_balls):
        x_avg = 0
        y_avg = 0
        x_vel_avg = 0
        y_vel_avg = 0
        total_filter_pos_weight = 0
        total_filter_vel_weight = 0

        # This should never happen, but it's buggy right now
        if len(kalman_balls) == 0:
            print('WorldBall::ERROR::NoKalmanBalls?')
            self.x = x_avg
            self.y = y_avg
            self.x_vel = x_vel_avg
            self.y_vel = y_vel_avg

            self.pos = (self.x, self.y)
            self.vel = (self.x_vel, self.y_vel)

            return

        # Get x, y, x_vel, y_vel uncertantity, sqrt(covariance along diagonal)
        # get 1 / health of filter
        # Multiple l2 norm of pos (xy) / vel (xy) uncertantity by health
        # Bring to the -1.5ish power
        # This add these as the position and velocity uncertanty

        # Add all of the states from each camera multiplied by that l2 norm to the -1.5ish power
        # Divide by the total found in the first step

        for ball in kalman_balls:
            # Get the covariance of everything
            #   How well we can predict the next measurement
            cov = ball.filter.P_k_k
            x_cov     = cov.item((0,0))
            x_vel_cov = cov.item((1,1))
            y_cov     = cov.item((2,2))
            y_vel_cov = cov.item((3,3))

            # Std dev of each state value
            #   Lower std dev means we have a better idea of the true value
            x_uncer     = math.sqrt(x_cov)
            x_vel_uncer = math.sqrt(x_vel_cov)
            y_uncer     = math.sqrt(y_cov)
            y_vel_uncer = math.sqrt(y_vel_cov)

            # Inversly proportional to how many times this filter has had camera updates
            filter_uncertantity = 1 / ball.health

            # How good of position or velocity measurement in total
            pos_uncer = math.sqrt(x_uncer*x_uncer + y_uncer*y_uncer)
            vel_uncer = math.sqrt(x_vel_uncer*x_vel_uncer + y_vel_uncer*y_vel_uncer)

            # Combines the filters trust in the data with our trust in the filter
            # Inverts the results because we want a more certain filter to have a higher weight
            # Applies a slight non-linearity with the power operation so the difference between numbers is more pronounced
            filter_pos_weight = math.pow(pos_uncer * filter_uncertantity, -util.config.ball_merger_power)
            filter_vel_weight = math.pow(vel_uncer * filter_uncertantity, -util.config.ball_merger_power)

            # Apply the weighting to all the estimates
            state = ball.filter.x_k_k
            x_avg     += filter_pos_weight * state.item(0)
            x_vel_avg += filter_vel_weight * state.item(1)
            y_avg     += filter_pos_weight * state.item(2)
            y_vel_avg += filter_vel_weight * state.item(3)

            total_filter_pos_weight += filter_pos_weight
            total_filter_vel_weight += filter_vel_weight

        # These should be greater than zero always since it's basically a 1 over a vector magnitude
        if (total_filter_pos_weight <= 0 or total_filter_vel_weight <= 0):
            print('WorldBall::ERROR::WeightsAreLTZero')

        # Scale back to the normal values
        x_avg     /= total_filter_pos_weight
        x_vel_avg /= total_filter_vel_weight
        y_avg     /= total_filter_pos_weight
        y_vel_avg /= total_filter_vel_weight

        self.x = x_avg
        self.y = y_avg
        self.x_vel = x_vel_avg
        self.y_vel = y_vel_avg

        self.pos = (self.x, self.y)
        self.vel = (self.x_vel, self.y_vel)
