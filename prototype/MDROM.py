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

@dataclass
class UniformDistribution:
    """
    Uniform distribution parameters
    """
    r_mean: np.array     # prismatic leg center [m]
    r_delta: float       # prismatic leg delta range [m]
    theta_mean: np.array # revolute leg center [rad]
    theta_delta: float   # revolute leg delta range [rad]


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
        
    def dynamics(self, x_com, p_left, p_right, u, d):
        """
        Compute the dynamics, xdot = f(x, u)
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

        # Flight domain (F)
        if d == 'F':
            # compute the dynamics
            a_com = np.array([[0],  
                              [-g]])
            xdot = np.vstack((v_com, a_com))

        # Left leg stance domain (L)
        elif d == 'L':
            # compute the leg state
            r_vec_L = p_left - p_com
            r_norm_L = np.linalg.norm(r_vec_L)
            r_hat_L = r_vec_L / r_norm_L

            # get the control input
            v_L = u[0]
            u_L = k * (v_L - l0)

            # compute the dynamics
            a_left = -(r_hat_L/m) * (k * (l0 - r_norm_L) + b * (v_com.T @ r_hat_L) + u_L)
            a_com = a_left + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        # Right leg stance domain (R)
        elif d == 'R':
            # compute the leg state
            r_vec_R = p_right - p_com
            r_norm_R = np.linalg.norm(r_vec_R)
            r_hat_R = r_vec_R / r_norm_R

            # get the control input
            v_R = u[1]
            u_R = k * (v_R - l0)

            # compute the dynamics
            a_right = -(r_hat_R/m) * (k * (l0 - r_norm_R) + b * (v_com.T @ r_hat_R) + u_R)
            a_com = a_right + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        # Double stance domain (D)
        elif d == 'D':
            # compute the leg state
            r_vec_L = p_left - p_com
            r_vec_R = p_right - p_com
            r_norm_L = np.linalg.norm(r_vec_L)
            r_norm_R = np.linalg.norm(r_vec_R)
            r_hat_L = r_vec_L / r_norm_L
            r_hat_R = r_vec_R / r_norm_R

            # get the control input
            v_L = u[0]
            v_R = u[1]
            u_L = k * (v_L - l0)
            u_R = k * (v_R - l0)

            # compute the dynamics
            a_left = -(r_hat_L/m) * (k * (l0 - r_norm_L) + b * (v_com.T @ r_hat_L) + u_L)
            a_right = -(r_hat_R/m) * (k * (l0 - r_norm_R) + b * (v_com.T @ r_hat_R) + u_R)
            a_com = a_left + a_right + np.array([[0], [-g]])
            xdot = np.vstack((v_com, a_com))

        return xdot    

    # interpolate the control input
    def interpolate_control_input(self, t, T, U):
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
    def RK3_rollout(self, x0_com, p_left, p_right, U, D0):
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

        # initial conditions
        x_left, x_right = self.update_leg_state(x0_com, p_left, p_right, U[:, 0], D0)
        xt_left[:, 0] = x_left.reshape(4)
        xt_right[:, 0] = x_right.reshape(4)
        xt_com[:, 0] = x0_com.reshape(4) 

        # RK3 integration
        xk_com = x0_com
        dk = D0
        for i in range(0, N-1):
            # intermmidiate times
            tk = i * dt
            t1 = tk
            t2 = tk + dt/2
            t3 = tk + dt

            # get the intermmediate inputs
            u1 = self.interpolate_control_input(t1, Tu, U)
            u2 = self.interpolate_control_input(t2, Tu, U)
            u3 = self.interpolate_control_input(t3, Tu, U)

            # RK3 vector fields
            f1 = self.dynamics(xk_com, 
                               p_left, 
                               p_right, 
                               u1, 
                               dk)
            f2 = self.dynamics(xk_com + dt/2 * f1, 
                               p_left, 
                               p_right, 
                               u2, 
                               dk)
            f3 = self.dynamics(xk_com - dt*f1 + 2*dt*f2, 
                               p_left, 
                               p_right, 
                               u3, 
                               dk)

            # take the step and update all states
            xk_com = xk_com + (dt/6) * (f1 + 4*f2 + f3)
            x_left, x_right = self.update_leg_state(xk_com, p_left, p_right, u1, dk)

            # TODO: check if you hit a switching surface

            # store the states
            xt_com[:, i+1] = xk_com.reshape(4)
            xt_left[:, i+1] = x_left.reshape(4)
            xt_right[:, i+1] = x_right.reshape(4)

        return Tx, xt_com, xt_left, xt_right
    
    # update leg polar state
    def update_leg_state(self, x_com, p_left, p_right, u, D):
        """
        Update the leg state based on the COM state and control input
        """
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
            r_L = np.linalg.norm(r_vec_L)
            rdot_L = -v_com.T @ r_hat_L
            theta_L = -np.arctan2(r_x_L, -r_z_L)
            thetadot_L = (r_z_L * rdot_x_L - r_x_L * rdot_z_L) / r_L

            # TODO: apply the control input, U
            u = None

            # pack into state vectors
            x_left = np.array([[r_L],
                               [theta_L[0]],
                               [rdot_L[0][0]],
                               [thetadot_L[0]]])
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
            r_R = np.linalg.norm(r_vec_R)
            rdot_R = -v_com.T @ r_hat_R
            theta_R = -np.arctan2(r_x_R, -r_z_R)
            thetadot_R = (r_z_R * rdot_x_R - r_x_R * rdot_z_R) / r_R

            # TODO: apply the control input, U
            u = None

            # pack into state vectors
            x_left = np.array([[self.l0],
                               [0.0],
                               [0.0],
                               [0.0]])
            x_right = np.array([[r_R],
                                [theta_R[0]],
                                [rdot_R[0][0]],
                                [thetadot_R[0]]])

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
            r_L = np.linalg.norm(r_vec_L)
            r_R = np.linalg.norm(r_vec_R)
            r_hat_L = r_vec_L / r_L
            r_hat_R = r_vec_R / r_R
            rdot_L = -v_com.T @ r_hat_L
            rdot_R = -v_com.T @ r_hat_R
            theta_L = -np.arctan2(r_x_L, -r_z_L)
            theta_R = -np.arctan2(r_x_R, -r_z_R)
            thetadot_L = (r_z_L * rdot_x_L - r_x_L * rdot_z_L) / r_L
            thetadot_R = (r_z_R * rdot_x_R - r_x_R * rdot_z_R) / r_R
            
            # pack into state vectors
            x_left = np.array([[r_L],
                            [theta_L[0]],
                            [rdot_L[0][0]],
                            [thetadot_L[0]]])
            x_right = np.array([[r_R],
                                [theta_R[0]],
                                [rdot_R[0][0]],
                                [thetadot_R[0]]])
            
        return x_left, x_right

    # Touch-Down (TD) Switching Surface -- checks individual legs
    def S_TD(self, x_com, x_leg):
        
        # get relevant states
        pz = x_com[1]
        vz = x_com[3]
        r = x_leg[0]
        theta = x_leg[1]
        rdot = x_leg[2]
        thetadot = x_leg[3]

        # compute the foot position and velocity
        pz_foot = pz - r * np.cos(theta)
        vz_foot = vz - rdot * np.cos(theta) + r * thetadot * np.sin(theta)

        # check the switching surface conditions
        gnd_pos = (pz_foot <= 0.0)      # foot is touching the ground or below
        neg_vel = (vz_foot <= 0.0)      # foot is moving downward
        touchdown = gnd_pos and neg_vel # if true, foot has touched the ground

        return touchdown

    # Take-Off (TO) Switching Surface -- checks individual legs
    def S_TO(self, x_com, x_leg):

        # get relevant states
        r = x_leg[0]
        rdot = x_leg[2]

        # check the switching surface conditions
        nom_length = (r >= self.l0)      # the leg is at its nominal uncompressed length
        pos_vel = (rdot >= 0.0)          # the leg is going in uncompressing direction
        takeoff = nom_length and pos_vel # if true, leg has taken off into flight

        return takeoff

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
    control_params = PredictiveControlParams(N=300, 
                                             dt=0.01, 
                                             K=100,
                                             interp='Z')

    # declare the uniform distribution
    # r_mean = np.ones(control_params.N-1) * system_params.l0
    r_mean = system_params.l0
    r_delta = 0.0
    # thtea_mean = np.ones(control_params.N-1) * 0.0
    theta_mean = 0.0
    theta_delta = 0.0
    distr_params = UniformDistribution(r_mean = r_mean,
                                       r_delta = r_delta,
                                       theta_mean = theta_mean,
                                       theta_delta = theta_delta)

    # declare reduced order model object
    mdrom = MDROM(system_params, control_params)

    # initial conditions
    x0_com = np.array([[0.25], # px [m]
                       [2.75], # py [m]
                       [1],  # vx [m/s]
                       [0]]) # vz [m/s]
    p_left = np.array([[0.0],  # px [m]
                       [0.0]]) # py [m]
    p_right = np.array([[0.5],  # px [m]
                        [0.0]]) # py [m]
    D0 = 'F'

    # control inputs
    U_r = np.random.uniform(distr_params.r_mean - distr_params.r_delta,
                            distr_params.r_mean + distr_params.r_delta,
                            (2, control_params.N-1))
    U_legs = np.random.uniform(distr_params.theta_mean - distr_params.theta_delta,
                               distr_params.theta_mean + distr_params.theta_delta,
                               (2, control_params.N-1))
    # U = np.ones((4, control_params.N-1)) * 0.65
    U = np.vstack((U_r, U_legs))

    # run the simulation
    t0 = time.time()
    t, x_com, x_left, x_right = mdrom.RK3_rollout(x0_com, p_left, p_right, U, D0)
    tf = time.time()
    print('Simulation time: ', tf - t0)

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
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, x_left[1, :], label='left')
    plt.plot(t, x_right[1, :], label='right')
    plt.xlabel('time [s]')
    plt.ylabel('theta [rad]')
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t, x_left[2, :], label='left')
    plt.plot(t, x_right[2, :], label='right')
    plt.xlabel('time [s]')
    plt.ylabel('rdot [m/s]')
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t, x_left[3, :], label='left')
    plt.plot(t, x_right[3, :], label='right')
    plt.xlabel('time [s]')
    plt.ylabel('thetadot [rad/s]')
    plt.grid()
    plt.legend()
    
    plt.show()
