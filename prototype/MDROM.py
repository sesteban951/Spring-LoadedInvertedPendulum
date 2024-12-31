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
        pt_left = np.zeros((2, N))
        pt_right = np.zeros((2, N))
        D = [None for _ in range(N)]

        # initial conditions
        x_left, x_right = self.update_leg_state(x0_com, p_left, p_right, U[:, 0], D0)
        xt_left[:, 0] = x_left.reshape(4)
        xt_right[:, 0] = x_right.reshape(4)
        xt_com[:, 0] = x0_com.reshape(4) 
        pt_left[:, 0] = p_left.reshape(2)
        pt_right[:, 0] = p_right.reshape(2)
        D[0] = D0

        # RK3 integration
        xk_com = x0_com
        Dk = D0
        contacts = self.contact_identification(Dk)
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
                               Dk)
            f2 = self.dynamics(xk_com + dt/2 * f1, 
                               p_left, 
                               p_right, 
                               u2, 
                               Dk)
            f3 = self.dynamics(xk_com - dt*f1 + 2*dt*f2, 
                               p_left, 
                               p_right, 
                               u3, 
                               Dk)

            # take the step and update all states
            xk_com = xk_com + (dt/6) * (f1 + 4*f2 + f3)
            x_left, x_right = self.update_leg_state(xk_com, p_left, p_right, u1, Dk)

            # check if hit switching surfaces and update state needed
            checked_contacts = self.check_switching(xk_com, x_left, x_right, contacts)
            if checked_contacts != contacts:
                
                # update the feet positions
                p_left, p_right = self.apply_reset(xk_com, x_left, x_right, contacts, checked_contacts)
                
                # update domain and contact info
                Dk = self.domain_identification(checked_contacts)
                contacts = checked_contacts

            # store the states
            xt_com[:, i+1] = xk_com.reshape(4)
            xt_left[:, i+1] = x_left.reshape(4)
            xt_right[:, i+1] = x_right.reshape(4)
            pt_left[:, i+1] = p_left.reshape(2)
            pt_right[:, i+1] = p_right.reshape(2)
            D[i+1] = Dk

        return Tx, xt_com, xt_left, xt_right, pt_left, pt_right, D
    
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

            # TODO: also get the velocity from the control input
            r_L = u[0]
            r_R = u[1]
            theta_L = u[2]
            theta_R = u[3]

            # pack into state vectors
            x_left = np.array([[r_L],
                               [theta_L],
                               [0.0],
                               [0.0]])
            x_right = np.array([[r_R],
                                [theta_R],
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

            # TODO: also get the velocity from the control input
            r_R = u[1]
            theta_R = u[3]

            # pack into state vectors
            x_left = np.array([[r_L],
                               [theta_L[0]],
                               [rdot_L[0][0]],
                               [thetadot_L[0]]])
            x_right = np.array([[r_R],
                                [theta_R],
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

            # TODO: also get the velocity from the control input
            r_L = u[0]
            theta_L = u[2]

            # pack into state vectors
            x_left = np.array([[r_L],
                               [theta_L],
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

    # check switching surfaces
    def check_switching(self, x_com, x_left, x_right, contacts):
        """
        Check if the state has hit any switching surfaces
        """
        # unpack the contact values
        contact_L = contacts[0]
        contact_R = contacts[1]

        # Left Leg
        if contact_L == False:
            # check for touchdown
            contact_result_L = self.S_TD(x_com, x_left)
        elif contact_L == True:
            # check for takeoff
            contact_result_L = self.S_TO(x_com, x_left)

        # Right Leg
        if contact_R == False:
            # check for touchdown
            contact_result_R = self.S_TD(x_com, x_right)
        elif contact_R == True:
            # check for takeoff
            contact_result_R = self.S_TO(x_com, x_right)

        # update the contact values
        contacts = [contact_result_L, contact_result_R]

        return contacts

    # Touch-Down (TD) Switching Surface -- checks individual legs
    def S_TD(self, x_com, x_leg):
        """
        Check if a leg has touched the ground
        """
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

        # return the contact result
        if touchdown==True:
            contact = True
        elif touchdown==False:
            contact = False

        return contact

    # Take-Off (TO) Switching Surface -- checks individual legs
    def S_TO(self, x_com, x_leg):
        """
        Check if a leg has taken off from the ground
        """
        # get relevant states
        r = x_leg[0]
        rdot = x_leg[2]

        # check the switching surface conditions
        nom_length = (r >= self.l0)      # the leg is at its nominal uncompressed length
        pos_vel = (rdot >= 0.0)          # the leg is going in uncompressing direction
        takeoff = nom_length and pos_vel # if true, leg has taken off into flight

        # return the contact result
        if takeoff==True:
            contact = False
        elif takeoff==False:
            contact = True

        return contact

    # apply a reset map
    def apply_reset(self, x_com, x_left, x_right, contacts_prev, contacts_new):

        # unapck the contact values
        contact_prev_L = contacts_prev[0]
        contact_prev_R = contacts_prev[1]
        contact_new_L = contacts_new[0]
        contact_new_R = contacts_new[1]

        # unpack the states
        p_com = np.array([[x_com[0]],
                          [x_com[1]]]).reshape(2, 1)
        
        # Left leg update
        if contact_prev_L != contact_new_L:
            # get the leg state
            r_L = x_left[0][0]
            theta_L = x_left[1][0]
            
            # TODO: right now this is a rough estimate consider better estimate
            px_left_COM = -r_L * np.sin(theta_L)
            px_left = p_com[0][0] + px_left_COM
            pz_left = 0.0  

            # update the left leg
            p_left = np.array([[px_left],[pz_left]])
        else:
            p_left = np.array([[None],
                               [None]])

        # Right leg update
        if contact_prev_R != contact_new_R:
            # get the leg state
            r_R = x_right[0][0]
            theta_R = x_right[1][0]

            # TODO: right now this is a rough estimate consider better estimate
            px_right_COM = -r_R * np.sin(theta_R)
            px_right = p_com[0][0] + px_right_COM
            pz_right = 0.0

            # update the right leg
            p_right = np.array([[px_right],[pz_right]])
        else:
            p_right = np.array([[None],
                                [None]])

        return p_left, p_right
    
    # Domain mapping function
    def domain_identification(self, contacts):
        """
        Map the contact state to a unique domain
        """
        # unpack the contact values
        contact_L = contacts[0]
        contact_R = contacts[1]

        # Flight (F)
        if (contact_L == False) and (contact_R == False):
            D = 'F'
        # Left leg support (L)
        elif (contact_L == True) and (contact_R == False):
            D = 'L'
        # Right leg support (R)
        elif (contact_L == False) and (contact_R == True):
            D = 'R'
        # Double leg support (D)
        elif (contact_L == True) and (contact_R == True):
            D = 'D'
        
        return D
    
    # Domain mapping function
    def contact_identification(self, D):
        """
        Simple contact identification
        """
        # Flight (F)
        if D == 'F':
            contacts = [False, False]
        # Left leg support (L)
        elif D == 'L':
            contacts = [True, False]
        # Right leg support (R)
        elif D == 'R':
            contacts = [False, True]
        # Double leg support (D)
        elif D == 'D':
            contacts = [True, True]
        
        return contacts

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
                       [1]]) # vz [m/s]
    p_left = np.array([[None],  # px [m]
                       [None]]) # py [m]
    p_right = np.array([[None],  # px [m]
                        [None]]) # py [m]
    D0 = 'F'

    # random control inputs
    # U_r = np.random.uniform(distr_params.r_mean - distr_params.r_delta,
    #                         distr_params.r_mean + distr_params.r_delta,
    #                         (2, control_params.N-1))
    # U_legs = np.random.uniform(distr_params.theta_mean - distr_params.theta_delta,
    #                            distr_params.theta_mean + distr_params.theta_delta,
    #                            (2, control_params.N-1))
    # U = np.vstack((U_r, U_legs))

    # constant fixed control inputs
    u_constant = np.array([[system_params.l0 *1.0], # left leg
                           [system_params.l0*1.0], # right leg
                           [np.pi/6],   # left leg
                           [-np.pi/6]]) # right leg
    U = np.tile(u_constant, (1, control_params.N-1))

    # run the simulation
    t0 = time.time()
    t, x_com, x_left, x_right, p_left, p_right, D = mdrom.RK3_rollout(x0_com, p_left, p_right, U, D0)
    tf = time.time()
    print('Simulation time: ', tf - t0)

    print(t.shape)
    print(x_com.shape)
    print(x_left.shape)
    print(x_right.shape)
    print(p_left.shape)
    print(p_right.shape)
    print(len(D))

    # plot the center of mass trajcetory
    plt.figure()

    # draw straight black line to represent the ground
    plt.plot([-1, 1], [0, 0], 'k-')

    plt.plot(x_com[0, :], x_com[1, :], label='x')
    plt.plot(x0_com[0], x0_com[1], 'go', label='x0')
    plt.plot(x_com[0, -1], x_com[1, -1], 'rx', label='xf')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.axis('equal') 
    plt.legend()
    plt.show()

    # # plot the leg states
    # plt.figure()

    # plt.subplot(2, 2, 1)
    # plt.plot(t, x_left[0, :], label='left')
    # plt.plot(t, x_right[0, :], label='right')
    # plt.xlabel('time [s]')
    # plt.ylabel('r [m]')
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 2, 2)
    # plt.plot(t, x_left[1, :], label='left')
    # plt.plot(t, x_right[1, :], label='right')
    # plt.xlabel('time [s]')
    # plt.ylabel('theta [rad]')
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 2, 3)
    # plt.plot(t, x_left[2, :], label='left')
    # plt.plot(t, x_right[2, :], label='right')
    # plt.xlabel('time [s]')
    # plt.ylabel('rdot [m/s]')
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 2, 4)
    # plt.plot(t, x_left[3, :], label='left')
    # plt.plot(t, x_right[3, :], label='right')
    # plt.xlabel('time [s]')
    # plt.ylabel('thetadot [rad/s]')
    # plt.grid()
    # plt.legend()
    
    # plt.show()

    # 805-994-8429
    # Crystal 
    # 25$