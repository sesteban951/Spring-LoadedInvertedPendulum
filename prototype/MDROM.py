#######################################################################
# Multidomain Reduced Order Model (MDROM) simulation
#######################################################################

import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

#######################################################################
# DATA CLASSES
#######################################################################

@dataclass
class SystemParams:
    """
    System parameters
    """
    m: float    # mass [kg]
    g: float    # gravity [m/s^2]
    l0: float   # spring free length [m]
    k: float    # spring stiffness [N/m]
    b: float    # damping coefficient [Ns/m]

@dataclass
class PredictiveControlParams:
    """
    Predictive control parameters
    """
    N: int      # prediction horizon
    dt: float   # time step [s]
    K: int      # number of rollouts

#######################################################################
# DYNMICS
#######################################################################

class MDROM:
    """
     Planar Spring loaded inverted pendulum (SLIP) model
    """

    def __init__(self, system_params, control_params):
        
        # initialize system parameters
        self.m = system_params.m
        self.g = system_params.g
        self.l0 = system_params.l0
        self.k = system_params.k
        self.b = system_params.b
        self.dt = control_params.dt
        self.N = control_params.N
        
    def dynamics(self, x_com, x_left, x_right, u, d):
        """
        Compute the dynamics of the system
        """
        # access the system parameters
        m = self.m
        g = self.g
        l0 = self.l0
        k = self.k
        b = self.b

        # unpack the state
        p_com = np.array([[x_com[0]],  # position of the center of mass
                          [x_com[1]]]).reshape(2, 1)
        v_com = np.array([[x_com[2]],  # velocity of the center of mass
                          [x_com[3]]]).reshape(2, 1)
        
        # unpack the leg states
        p_left = np.array([[x_left[0]],  # position of the left leg
                           [x_left[1]]]).reshape(2, 1)
        v_left = np.array([[x_left[2]],  # velocity of the left leg
                           [x_left[3]]]).reshape(2, 1)
        p_right = np.array([[x_right[0]],  # position of the right leg
                            [x_right[1]]]).reshape(2, 1)
        v_right = np.array([[x_right[2]],  # velocity of the right leg
                            [x_right[3]]]).reshape(2, 1)
        
        # Flight domain (F)
        if d == 'F':
            # compute the dynamics
            a_com = np.array([[0],  
                              [-g]])
            xdot = np.vstack((v_com, a_com))

        # Left leg stance domain (L)
        elif d == 'L':
            # compute the leg state
            rL = p_com - p_left
            rL_norm = np.linalg.norm(rL)
            rL_hat = rL / rL_norm

            # get the control input
            vL = u[0]
            uL = k * (vL - l0)

            # compute the dynamics
            a_com = rL_hat * ((k/m) * (l0 - rL_norm) - (b/m) * (v_com.T @ rL) / rL_norm + (1/m) * uL) + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        # Right leg stance domain (R)
        elif d == 'R':
            # compute the leg state
            rR = p_com - p_right
            rR_norm = np.linalg.norm(rR)
            rR_hat = rR / rR_norm

            # get the control input
            vR = u[1]
            uR = k * (vR - l0)

            # compute the dynamics
            a_com = rR_hat * ((k/m) * (l0 - rR_norm) - (b/m) * (v_com.T @ rR) / rR_norm + (1/m) * uR) + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        # Double stance domain (D)
        elif d == 'D':
            # compute the leg state
            rL = p_com - p_left
            rR = p_com - p_right
            rL_norm = np.linalg.norm(rL)
            rR_norm = np.linalg.norm(rR)
            rL_hat = rL / rL_norm
            rR_hat = rR / rR_norm

            # get the control input
            vL = u[0]
            vR = u[1]
            uL = k * (vL - l0)
            uR = k * (vR - l0)

            # compute the dynamics
            a_left = rL_hat * ((k/m) * (l0 - rL_norm) - (b/m) * (v_com.T @ rL) / rL_norm + (1/m) * uL)
            a_right = rR_hat * ((k/m) * (l0 - rR_norm) - (b/m) * (v_com.T @ rR) / rR_norm + (1/m) * uR)
            a_com = a_left + a_right + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))
        
        return xdot
    
    # def RK3 integration scheme
    def RK3_rollout(self, x0_com, x0_left, x0_right, U, D0):
        """
        Runge-Kutta 3rd order integration scheme
        """
        # integration parameters
        dt = self.dt
        N = self.N

        # make the containers
        Tx = np.arange(0, N) * dt
        Tu = np.arange(0, N-1) * dt
        xt_com = np.zeros((4, N))

        xt_com[:, 0] = x0_com.reshape(4) 

        # RK3 integration
        xk_com = x0_com
        xk_left = x0_left
        xk_right = x0_right
        dk = D0
        t = 0
        for i in range(0, N-1):
            # intermmidiate times
            t1 = t
            t2 = t + dt/2
            t3 = t + dt

            # get the intermmediate inputs
            u1 = self.get_control_input(t1, Tu, U)
            u2 = self.get_control_input(t2, Tu, U)
            u3 = self.get_control_input(t3, Tu, U)

            # RK3 vector fields
            f1 = self.dynamics(xk_com, 
                               xk_left, 
                               xk_right, 
                               u1, 
                               dk)
            f2 = self.dynamics(xk_com + dt/2 * f1, 
                               xk_left, 
                               xk_right, 
                               u2, 
                               dk)
            f3 = self.dynamics(xk_com - dt*f1 + 2*dt*f2, 
                               xk_left, 
                               xk_right, 
                               u3, 
                               dk)

            # take the step
            xk_com = xk_com + (dt/6) * (f1 + 4*f2 + f3)
            xt_com[:, i+1] = xk_com.reshape(4)

        return Tx, xt_com
    
    def get_control_input(self, t, Tu, U):
        """
        Get the control input
        """
        # find which interval the time belongs to
        idx = np.where(Tu <= t)[0][-1]

        u = np.zeros((2, 1))

        return u
    
#######################################################################
# DISTRIBUTION
#######################################################################

class ParametricDistribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":

    # decalre the system parameters
    system_params = SystemParams(m=1.0, 
                                 g=9.81, 
                                 l0=1.0, 
                                 k=100.0, 
                                 b=10.0)
    
    # declare control parameters
    control_params = PredictiveControlParams(N=10, 
                                             dt=0.01, 
                                             K=100)
    
    # declare reduced order model object
    mdrom = MDROM(system_params, control_params)

    # initial conditions
    x0_com = np.array([[0.0], 
                       [1.0], 
                       [1.0], 
                       [0.0]])
    x0_left = np.array([[0.0], 
                        [0.0], 
                        [0.0], 
                        [0.0]])
    x0_right = np.array([[0.0], 
                        [0.0], 
                        [0.0], 
                        [0.0]])
    D0 = 'F'
    U = np.zeros((2, control_params.N-1))

    # run the simulation
    t, x = mdrom.RK3_rollout(x0_com, x0_left, x0_right, U, D0)
    
    # plt.figure()
    # plt.plot(x[0, :], x[1, :], label='x')
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.grid()
    # plt.legend()
    # plt.show()