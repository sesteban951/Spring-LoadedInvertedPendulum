#######################################################################
# Multidomain Reduced Order Model (MDROM) simulation
#######################################################################

import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import time

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
    interp: str # interpolation method, 'Z' for zero order hold, 'L' for linear

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

        # initialize control parameters
        self.dt = control_params.dt
        self.N = control_params.N
        self.interp = control_params.interp
        
    def dynamics(self, x_com, x_left, x_right, u, d):
        """
        Compute hte dynamics, xdot = f(x, u)
            x_com: state of the center of mass in world frame
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
            rL = p_left - p_com
            rL_norm = np.linalg.norm(rL)
            rL_hat = rL / rL_norm

            # get the control input
            vL = u[0]
            uL = k * (vL - l0)

            # compute the dynamics
            a_left = -(rL_hat/m) * (k * (l0 - rL_norm) + b * (v_com.T @ rL_hat) + uL)
            a_com = a_left + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        # Right leg stance domain (R)
        elif d == 'R':
            # compute the leg state
            rR = p_right - p_com
            rR_norm = np.linalg.norm(rR)
            rR_hat = rR / rR_norm

            # get the control input
            vR = u[1]
            uR = k * (vR - l0)

            # compute the dynamics
            a_right = -(rR_hat/m) * (k * (l0 - rR_norm) + b * (v_com.T @ rR_hat) + uR)
            a_com = a_right + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        # Double stance domain (D)
        elif d == 'D':
            # compute the leg state
            rL = p_left - p_com
            rR = p_right - p_com
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
            a_left =  -(rL_hat/m) * (k * (l0 - rL_norm) + b * (v_com.T @ rL_hat) + uL)
            a_right = -(rR_hat/m) * (k * (l0 - rR_norm) + b * (v_com.T @ rR_hat) + uR)
            a_com = a_left + a_right + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))
        
        # get the polar leg states
        # x_left, x_right = self.update_leg_state(x_com, p_left, p_right, u, d)
        # print('x_left: ', x_left)
        # print('x_right: ', x_right)

        return xdot    

    # interpolate the control input
    def get_control_input(self, t, T, U):
        """
        Interpolate the control input signal. 
        """
        # find which interval the time belongs to
        idx = np.where(T <= t)[0][-1]

        # zero order hold interpolation
        if self.interp == 'Z':
            # constant control input
            u = U[:, idx]
        
        # linear interpolation
        elif self.interp == 'L':
            # beyond the last knot
            if idx == len(T) - 1:
                u = U[:, idx]

            # within an interval
            else:
                # knot ends
                t0 = T[idx]
                tf = T[idx+1]
                u0 = U[:, idx]
                uf = U[:, idx+1]

                # linear interpolation
                u = u0 + (uf - u0) * (t - t0) / (tf - t0)

        return u
    
    # def RK3 integration scheme
    def RK3_rollout(self, x0_com, x0_left, x0_right, p0_feet, U, D0):
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
        xt_left = np.zeros((4, N))
        xt_right = np.zeros((4, N))
        xt_com[:, 0] = x0_com.reshape(4) 

        # get the feet positions
        p_left = np.array([[x0_left[0]],  # position of the left leg
                           [x0_left[1]]]).reshape(2, 1)
        p_right = np.array([[x0_right[0]],  # position of the right leg
                            [x0_right[1]]]).reshape(2, 1)

        # RK3 integration
        xk_com = x0_com
        xk_left = x0_left
        xk_right = x0_right
        dk = D0
        for i in range(0, N-1):
            # intermmidiate times
            tk = i * dt
            t1 = tk
            t2 = tk + dt/2
            t3 = tk + dt

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
            x_left, x_right = self.update_leg_state(xk_com, p_left, p_right, u1, dk)

            # store the states
            xt_com[:, i+1] = xk_com.reshape(4)
            xt_left[:, i+1] = x_left.reshape(4)
            xt_right[:, i+1] = x_right.reshape(4)

        return Tx, xt_com, xt_left, xt_right
    
    # update leg polar state
    def update_leg_state(self, x_com, p_left, p_right, u, D):

        # unpack COM state
        p_com = np.array([[x_com[0]],  
                          [x_com[1]]]).reshape(2, 1)
        v_com = np.array([[x_com[2]],
                          [x_com[3]]]).reshape(2, 1)
        
        # in flight (state governed by kinemaitc input)
        if D == 'F':

            # TODO: apply the control input, U
            u = None

            # pack into state vectors
            x_left = np.array([[self.l0],
                               [0.0],
                               [0.0],
                               [0.0]])
            x_right = np.array([[self.l0],
                                [0.0],
                                [0.0],
                                [0.0]])
        
        # in left support (leg state governed by dynamics)
        if D == 'L':
            # compute relevant vectors
            r_vec_L = p_left - p_com
            r_hat_L = r_vec_L / np.linalg.norm(r_vec_L)
            rdot_vec_L = -v_com

            r_x_L = r_vec_L[0]
            r_z_L = r_vec_L[1]
            rdot_x_L = rdot_vec_L[0]
            rdot_z_L = rdot_vec_L[1]

            # compute the left leg state in polar coordinates
            rL = np.linalg.norm(r_vec_L)
            rdot_L = -v_com.T @ r_hat_L
            theta_L = -np.arctan2(r_x_L, -r_z_L)
            thetadot_L = (r_z_L * rdot_x_L - r_x_L * rdot_z_L) / rL

            # TODO: apply the control input, U
            u = None

            # pack into state vectors
            x_left = np.array([[rL],
                               [theta_L],
                               [rdot_L],
                               [thetadot_L]])
            x_right = np.array([[self.l0],
                                [0.0],
                                [0.0],
                                [0.0]])

        # in right support (leg state governed by dynamics)
        if D == 'R':
            # compute relevant vectors
            r_vec_R = p_right - p_com
            r_hat_R = r_vec_R / np.linalg.norm(r_vec_R)
            rdot_vec_R = -v_com

            r_x_R = r_vec_R[0]
            r_z_R = r_vec_R[1]
            rdot_x_R = rdot_vec_R[0]
            rdot_z_R = rdot_vec_R[1]

            # compute the right leg state in polar coordinates
            rR = np.linalg.norm(r_vec_R)
            rdot_R = -v_com.T @ r_hat_R
            theta_R = -np.arctan2(r_x_R, -r_z_R)
            thetadot_R = (r_z_R * rdot_x_R - r_x_R * rdot_z_R) / rR

            # TODO: apply the control input, U
            u = None

            # pack into state vectors
            x_left = np.array([[self.l0],
                               [0.0],
                               [0.0],
                               [0.0]])
            x_right = np.array([[rR],
                                [theta_R],
                                [rdot_R],
                                [thetadot_R]])

        # in double support (leg state governed by dynamics)
        if D == 'D':
            # compute relevant vectors
            r_vec_L = p_left - p_com
            r_vec_R = p_right - p_com

            rdot_vec_L = -v_com  # assuming foot is pinned and not slipping
            rdot_vec_R = -v_com  # assuming foot is pinned and not slipping

            r_x_L = r_vec_L[0]
            r_z_L = r_vec_L[1]
            r_x_R = r_vec_R[0]
            r_z_R = r_vec_R[1]
            rdot_x_L = rdot_vec_L[0]
            rdot_z_L = rdot_vec_L[1]
            rdot_x_R = rdot_vec_R[0]
            rdot_z_R = rdot_vec_R[1]

            # compute the leg state in polar coordinates
            rL = np.linalg.norm(r_vec_L)
            rR = np.linalg.norm(r_vec_R)
            r_hat_L = r_vec_L / rL
            r_hat_R = r_vec_R / rR
            rdot_L = -v_com.T @ r_hat_L
            rdot_R = -v_com.T @ r_hat_R
            theta_L = -np.arctan2(r_x_L, -r_z_L)
            theta_R = -np.arctan2(r_x_R, -r_z_R)
            thetadot_L = (r_z_L * rdot_x_L - r_x_L * rdot_z_L) / rL
            thetadot_R = (r_z_R * rdot_x_R - r_x_R * rdot_z_R) / rR
            
            # pack into state vectors
            x_left = np.array([[rL],
                            [theta_L[0]],
                            [rdot_L[0][0]],
                            [thetadot_L[0]]])
            x_right = np.array([[rR],
                                [theta_R[0]],
                                [rdot_R[0][0]],
                                [thetadot_R[0]]])
            
        return x_left, x_right

#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":

    # decalre the system parameters
    system_params = SystemParams(m=35.0, 
                                 g=9.81, 
                                 l0=0.65, 
                                 k=5000.0, 
                                 b=500.0)
    
    # declare control parameters
    control_params = PredictiveControlParams(N=500, 
                                             dt=0.01, 
                                             K=100,
                                             interp='Z')
    
    # declare reduced order model object
    mdrom = MDROM(system_params, control_params)

    # initial conditions
    x0_com = np.array([[0.25], # px [m]
                       [0.75], # py [m]
                       [0],  # vx [m/s]
                       [0]]) # vy [m/s]
    x0_left = np.array([[0.0],  # r [m]
                        [0.0], # theta[rad]
                        [0.0],  # rdot [m/s]
                        [0.0]]) # thetadot [rad/s]
    x0_right = np.array([[0.5], 
                        [0.0], 
                        [0.0], 
                        [0.0]])
    p0_feet = np.array([[0.0, 0.0], 
                        [0.5, 0.0]])
    D0 = 'D'

    # control inputs
    U = np.ones((2, control_params.N-1)) * 0.65

    # run the simulation
    t0 = time.time()
    t, x_com, x_left, x_right = mdrom.RK3_rollout(x0_com, x0_left, x0_right, p0_feet, U, D0)
    tf = time.time()
    print('Simulation time: ', tf - t0)

    # # # run the simulation
    # sims = 500
    # DT = []
    # for i in range(0, sims):
    #     t0 = time.time()
    #     t, x = mdrom.RK3_rollout(x0_com, x0_left, x0_right, p0_feet, U, D0)
    #     tf = time.time()
    #     dt = tf - t0
    #     print('Iteration: ', i) 
    #     print('Simulation time: ', dt)
    #     DT.append(dt)
    # DT = np.array(DT)
    # print('Average simulation time: ', np.mean(DT))
    # print('Frequency: ', np.mean(1/DT))
    
    # plot the center of mass trajcetory
    plt.figure()
    plt.plot(0, 0, 'ko', label='ground')    
    plt.plot(x_com[0, :], x_com[1, :], label='x')
    plt.plot(x0_com[0], x0_com[1], 'go', label='x0')
    plt.plot(x_com[0, -1], x_com[1, -1], 'rx', label='xf')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.axis('equal') 
    plt.legend()
    plt.show()

    # plot the leg states
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(t, x_left[0, :], label='left')
    plt.plot(t, x_right[0, :], label='right')
    plt.xlabel('time [s]')
    plt.ylabel('r [m]')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, x_left[1, :], label='left')
    plt.plot(t, x_right[1, :], label='right')
    plt.xlabel('time [s]')
    plt.ylabel('theta [rad]')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t, x_left[2, :], label='left')
    plt.plot(t, x_right[2, :], label='right')
    plt.xlabel('time [s]')
    plt.ylabel('rdot [m/s]')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t, x_left[3, :], label='left')
    plt.plot(t, x_right[3, :], label='right')
    plt.xlabel('time [s]')
    plt.ylabel('thetadot [rad/s]')
    plt.legend()
    
    plt.show()
