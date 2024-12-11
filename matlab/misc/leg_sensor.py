#############################################################################
# Leg Sensing
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
import time

class LegSensor():
    """
    Simple class detect all switching events for a particular leg
    """
    def __init__(self):
        
        # states
        self.x_com = None   # COM state: x, z, xdot, zdot (in world frame)
        self.x_leg = None   # leg state: r, theta, rdot, thetadot (local to COM frame)
        self.x_foot = None  # foot state: x, z, xdot, zdot (in world frame) 

        # domain
        self.D = None       # domain of the leg (flight, ground)

    # update the states
    def update_state(self, x_com, x_leg):
        
        # update the states
        self.x_com = x_com
        self.x_leg = x_leg

        # com state
        px_com = x_com[0]
        pz_com = x_com[1]
        vx_com = x_com[2]
        vz_com = x_com[3]

        # leg state
        r = x_leg[0]
        theta = x_leg[1]
        rdot = x_leg[2]
        thetadot = x_leg[3]

        # foot position (world frame)
        px_foot = px_com - r * np.sin(theta)
        pz_foot = pz_com - r * np.cos(theta)

        # foot velocity (world frame)
        vx_foot = vx_com - rdot * np.sin(theta) - r * thetadot * np.cos(theta)
        vz_foot = vz_com - rdot * np.cos(theta) + r * thetadot * np.sin(theta)

        # update the foot state
        self.x_foot = np.array([px_foot, pz_foot, vx_foot, vz_foot])

class DynamicsIntegrator():
    """
    Simple class to simulate dynamics.
    """
    def __init__(self, t0, tf, dt):
        
        # dynamics info
        self.t0 = t0
        self.tf = tf
        self.dt = dt

    # RK2 method given the t, x0,and f(x)
    def RK2_step(self, t, xk, uk, f):

        # take an integration step using RK2 scheme
        f1 = f(t, xk, uk)
        f2 = f(t, xk + 0.5 * self.dt * f1, uk)
        xk_next = xk + self.dt * f2

        return xk_next

    # flight dynamics
    def flight_dynamics(self, t, x, u):
        # unpack the state
        vx = x[2]
        vz = x[3]

        # return f(x,u)
        xdot = np.array([vx, 
                         vz, 
                         0, 
                         -9.81])
        return xdot

    # compute the parabolic trajectory
    def parabolic_traj(self, x0):

        # create a time span
        times = np.arange(self.t0, self.tf, self.dt)
        
        # initial conditions
        px_0 = x0[0]
        pz_0 = x0[1]
        vx_0 = x0[2]
        vz_0 = x0[3]

        # create the trajectory
        x = np.zeros((len(times), len(x0)))

        # compute the trajectory
        for i in range(len(times)):
            t = times[i]
            x[i, 0] = px_0 + vx_0 * t
            x[i, 1] = pz_0 + vz_0 * t - 0.5 * 9.81 * t**2
            x[i, 2] = vx_0
            x[i, 3] = vz_0 - 9.81 * t

        return x

# just testing some stuff out
if __name__ == "__main__":

    # create the dynamics object
    t0 = 0.0
    tf = 1.0
    dt = 0.01
    d = DynamicsIntegrator(t0, tf, dt)

    # initial conditions
    x0_com = np.array([0,   # px
                       3,   # pz
                       0.5, # vx
                       0])  # vz

    # create the leg sensor
    l = LegSensor()
    x0_leg = np.array([1,    # r
                       0.15, # theta
                       0,    # rdot
                       0])   # thetadot

    # analytical flight
    x_t = d.parabolic_traj(x0_com)    

    # numerical flight
    t_span = np.arange(t0, tf, dt)
    
    X_com  = np.zeros((len(t_span), 4))
    X_leg  = np.zeros((len(t_span), 4))
    X_foot = np.zeros((len(t_span), 4))
    
    X_com[0, :] = x0_com
    xk_com = x0_com
    for i in range(len(t_span)):
        if i == len(t_span) - 1:
            break
        xk_com = d.RK2_step(0, xk_com, 0.0, d.flight_dynamics)
        X_com[i+1, :] = xk_com

    # plot the trajecotry
    plt.figure()
    plt.plot(x_t[:, 0], x_t[:, 1])
    plt.plot(X_com[:, 0], X_com[:, 1])
    plt.show()


