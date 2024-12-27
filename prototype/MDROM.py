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

        # states in world frame
        self.x_com = np.zeros((4, 1))    # center of mass state
        self.x_left = np.zeros((4, 1))   # left leg state
        self.x_right = np.zeros((4, 1))  # right leg state

        # initiliaze the domain
        self.domain = None # 'F': flight, 'L': left leg stance, 'R': right leg stance, 'D': double stance
        self.contact = [None, None]  # '[0,0]': flight, 
                                     # '[1,0]': left leg stance, 
                                     # '[0,1]': right leg stance, 
                                     # '[1,1]': double stance
        
    def dynamics(self, u):
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
        p_com = np.array([[self.x_com[0]],  # position of the center of mass
                          [self.x_com[1]]])
        v_com = np.array([[self.x_com[2]],  # velocity of the center of mass
                          [self.x_com[3]]])
        
        # Flight domain (F)
        if self.D == 'F':
            # compute the dynamics
            a_com = np.array([[0],  
                              [-g]])
            xdot = np.vstack((v_com, a_com))

        # Left leg stance domain (L)
        elif self.D == 'L':
            # compute the leg state
            rL = p_com - self.x_left[0:2]
            rL_norm = np.linalg.norm(rL)
            rL_hat = rL / rL_norm

            # get the control input
            vL = u[0]
            uL = k * (vL - l0)

            # compute the dynamics
            a_com = rL_hat * ((k/m) * (l0 - rL_norm) - (b/m) * (v_com.T @ rL) / rL_norm + (1/m) * uL) + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        # Right leg stance domain (R)
        elif self.D == 'R':
            # compute the leg state
            rR = p_com - self.x_right[0:2]
            rR_norm = np.linalg.norm(rR)
            rR_hat = rR / rR_norm

            # get the control input
            vR = u[1]
            uR = k * (vR - l0)

            # compute the dynamics
            a_com = rR_hat * ((k/m) * (l0 - rR_norm) - (b/m) * (v_com.T @ rR) / rR_norm + (1/m) * uR) + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        # Double stance domain (D)
        elif self.D == 'D':
            # compute the leg state
            rL = p_com - self.x_left[0:2]
            rR = p_com - self.x_right[0:2]
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


