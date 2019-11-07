from motor import GearMotor, Gearbox, Motor
import time
import numpy as np
import decimal

'''
This would be replaced by data from the real motor
The goal is to get these values out of our algorithm
'''
actualMotorSimulation = Motor(
        resistance=1.03,
        torque_constant=0.0335,
        speed_constant=0.0335,
        rotor_inertia=135*1e-3*(1e-2**2)
)
actualGearboxSimulation = Gearbox(
        gear_ratio=20.0 / 60.0,
        gear_inertia=0
)
realMotor = GearMotor(
        motor=actualMotorSimulation,
        gearbox=actualGearboxSimulation
)

dt = 0.01 #10 ms (arbitrary)
Vmax = 24.0 #Max voltage

angular_acceleration = float(0.0)
angular_velocity = float(0.0)

# Simulation of "real-life" motor
def simulateRealWorld(voltage):
    I = realMotor.inertia()
    Va = voltage #Applied voltage
    W = 0 #Current angular velocity
    A = 0 #Current angular acceleration
    Th = 0 #Current motor angle
    #Calculations
    T = realMotor.forward_dynamics(W,Va) #Torque
    A = T/I # Angular accerlation = torque/momment of inertia
    Th += W*dt + 0.5*A*(dt**2) #Angular version of: x = v*t + 1/2*a*t^2
    angular_acceleration = A
    angular_velocity = W

def calculate(voltage):
    #Apply the voltage and see what the output is
    simulateRealWorld(voltage)

def main():
    VStep = 0.5
    #Ramp up    
    for volts in np.arange(0, Vmax, VStep):
        calculate(volts)
        time.sleep(dt)

    #Cruise
    V = Vmax #Or some other voltage we will worry about later
    for _ in np.arange(0.0, 2.0, dt): #0 to 2 seconds
        calculate(V)
        time.sleep(dt)

    #Ramp down
    for volts in np.arange(Vmax, 0, VStep):
        calculate(volts)
        time.sleep(dt)

if __name__ == "__main__":
    main()