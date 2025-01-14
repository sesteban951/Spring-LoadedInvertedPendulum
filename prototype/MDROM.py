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
    m: float                # mass [kg]
    g: float                # gravity [m/s^2]
    k: float                # spring stiffness [N/m]
    b: float                # damping coefficient [Ns/m]
    l0: float               # spring free length [m]
    r_min: float            # leg min length [m]
    r_max: float            # leg max length [m]
    theta_min: float        # revolute leg min angle [rad]
    theta_max: float        # revolute leg max angle [rad]
    rdot_lim: float         # leg vel max velocity [m/s]
    thetadot_lim: float     # revolute leg max angular velocity [rad/s]
    torque_ankle: bool      # ankle torque enabled
    torque_ankle_lim: float # ankle torque limit [Nm]
    torque_ankle_kp: float  # ankle torque proportional gain
    torque_ankle_kd: float  # ankle torque derivative gain

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
# DYNAMICS
#######################################################################

class MDROM:
    """
     Planar Spring loaded inverted pendulum (SLIP) model
    """

    def __init__(self, system_params, control_params):
        
        # initialize system parameters
        self.m = system_params.m
        self.g = system_params.g
        self.k = system_params.k
        self.b = system_params.b
        self.l0 = system_params.l0
        self.lmin = system_params.lmin
        self.lmax = system_params.lmax

        # initialize control parameters
        self.dt = control_params.dt
        self.N = control_params.N
        self.interp = control_params.interp

    #################################  DYNAMICS  #################################

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
    
    #################################  ROLLOUT  #################################

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
        contacts = self.contact_identification(D0)
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

            # RK3 step and update all states
            xk_com = xk_com + (dt/6) * (f1 + 4*f2 + f3)
            xk_left, xk_right = self.update_leg_state(xk_com, p_left, p_right, u1, Dk)

            # check if hit switching surfaces and update state needed
            checked_contacts = self.check_switching(xk_com, xk_left, xk_right, u1, contacts)
            
            # if a change in contact detected
            if checked_contacts != contacts:
                
                # update the feet positions
                p_left, p_right = self.apply_reset(xk_com, xk_left, xk_right, checked_contacts)
                
                # update domain and contact info
                Dk = self.domain_identification(checked_contacts)
                contacts = checked_contacts

            # store the states
            xt_com[:, i+1] = xk_com.reshape(4)
            xt_left[:, i+1] = xk_left.reshape(4)
            xt_right[:, i+1] = xk_right.reshape(4)
            
            # store the foot positions
            if Dk == 'F':
                pt_left[:, i+1] = self.get_foot_pos(xk_com, xk_left, pz_zero=False).flatten()
                pt_right[:, i+1] = self.get_foot_pos(xk_com, xk_right, pz_zero=False).flatten()
            elif Dk == 'L':
                pt_left[:, i+1] = p_left.reshape(2)
                pt_right[:, i+1] = self.get_foot_pos(xk_com, xk_right, pz_zero=False).flatten()
            elif Dk == 'R':
                pt_left[:, i+1] = self.get_foot_pos(xk_com, xk_left, pz_zero=False).flatten()
                pt_right[:, i+1] = p_right.reshape(2)
            elif Dk == 'D':
                pt_left[:, i+1] = p_left.reshape(2)
                pt_right[:, i+1] = p_right.reshape(2)

            # store the domain
            D[i+1] = Dk

        return Tx, xt_com, xt_left, xt_right, pt_left, pt_right, D
    
    #################################  STATE UPDATES  #################################

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

    # compute foot position and veclocity in world
    def get_foot_pos(self, x_com, x_leg, pz_zero):

        # unpack the states
        p_com = np.array([[x_com[0]],
                          [x_com[1]]]).reshape(2, 1)
        
        # get the leg state
        r = x_leg[0][0]
        theta = x_leg[1][0]

        # foot position in world frame frame 
        px_foot_W = p_com[0][0] - r * np.sin(theta)
        
        if pz_zero == True:
            pz_foot_W = 0.0
        else:
            pz_foot_W = p_com[1][0] -r * np.cos(theta)
        
        p_foot_W = np.array([[px_foot_W],
                             [pz_foot_W]]).reshape(2, 1)

        return p_foot_W

    #################################  SWITCHING  #################################

    # check switching surfaces
    def check_switching(self, x_com, x_left, x_right, u, contacts):
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
            u_leg = u[0]
            contact_result_L = self.S_TO(x_com, x_left, u_leg)

        # Right Leg
        if contact_R == False:
            # check for touchdown
            contact_result_R = self.S_TD(x_com, x_right)
        elif contact_R == True:
            # check for takeoff
            u_leg = u[1]
            contact_result_R = self.S_TO(x_com, x_right, u_leg)

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
        r = x_leg[0]         # set by input
        theta = x_leg[1]     # set by input
        rdot = x_leg[2]      # set by input
        thetadot = x_leg[3]  # set by input
 
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
    def S_TO(self, x_com, x_leg, u_leg):
        """
        Check if a leg has taken off from the ground
        """
        # get relevant states
        r = x_leg[0]
        rdot = x_leg[2]

        # check the switching surface conditions
        # set by input
        nom_length = (r >= u_leg)      # the leg is at its nominal uncompressed length # TODO: change based on inp
        pos_vel = (rdot >= 0.0)          # the leg is going in uncompressing direction
        takeoff = nom_length and pos_vel # if true, leg has taken off into flight

        # return the contact result
        if takeoff==True:
            contact = False
        elif takeoff==False:
            contact = True

        return contact

    # apply a reset map
    def apply_reset(self, x_com, x_left, x_right, contacts_new):

        # unapck the contact values
        contact_new_L = contacts_new[0]
        contact_new_R = contacts_new[1]

        # Left leg update
        if contact_new_L == True:
            p_left = self.get_foot_pos(x_com, x_left, pz_zero=True)
        elif contact_new_L == False:
            p_left = np.array([[None],
                               [None]])

        # Right leg update
        if contact_new_R == True:
            p_right = self.get_foot_pos(x_com, x_right, pz_zero=True)
        elif contact_new_R == False:
            p_right = np.array([[None],
                                [None]])

        return p_left, p_right

    #################################  DOMAIN/CONTACT MAPPING  #################################
    
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
                                 k=5000.0, 
                                 b=500.0,
                                 l0=0.65,
                                 lmin=0.3,
                                 lmax=0.7)
    
    # declare control parameters
    control_params = PredictiveControlParams(N=150, 
                                             dt=0.01, 
                                             K=100,
                                             interp='L')

    # declare the uniform distribution
    # r_mean = np.ones(control_params.N-1) * system_params.l0
    # r_mean = system_params.l0
    # r_delta = 0.0
    # thtea_mean = np.ones(control_params.N-1) * 0.0
    # theta_mean = 0.0
    # theta_delta = 0.0
    # distr_params = UniformDistribution(r_mean = r_mean,
    #                                    r_delta = r_delta,
    #                                    theta_mean = theta_mean,
    #                                    theta_delta = theta_delta)

    # declare reduced order model object
    mdrom = MDROM(system_params, control_params)

    # initial conditions
    x0_com = np.array([[0.25], # px [m]
                       [0.50], # py [m]
                       [3],  # vx [m/s]
                       [0]]) # vz [m/s]
    p_left = np.array([[0],  # px [m]
                       [0]]) # py [m]
    p_right = np.array([[0.5],  # px [m]
                        [0]]) # py [m]
    D0 = 'D'

    # CONSTANT INPUT
    # u_constant = np.array([[system_params.l0 * 1.0], # left leg
    #                        [system_params.l0 * 1.0], # right leg
    #                        [np.pi/8],   # left leg
    #                        [-np.pi/8]]) # right leg
    # U = np.tile(u_constant, (1, control_params.N-1))

    # SPLINE INPUT
    f = 2.0
    omega = 2*np.pi*f
    A = 0.2
    u_left = A * np.sin(omega * np.arange(0, control_params.N-1) * control_params.dt)  + system_params.l0
    u_right = A * np.sin(omega * np.arange(0, control_params.N-1) * control_params.dt) + system_params.l0
    u_theta_left = np.ones(control_params.N-1) * np.pi/8
    u_theta_right = np.ones(control_params.N-1) * -np.pi/8
    U = np.vstack((u_left, u_right, u_theta_left, u_theta_right))
    # print(U.shape)  

    # RANDOM INPUT
    # U_r = np.random.uniform(distr_params.r_mean - distr_params.r_delta,
    #                         distr_params.r_mean + distr_params.r_delta,
    #                         (2, control_params.N-1))
    # U_legs = np.random.uniform(distr_params.theta_mean - distr_params.theta_delta,
    #                            distr_params.theta_mean + distr_params.theta_delta,
    #                            (2, control_params.N-1))
    # U = np.vstack((U_r, U_legs))

    # run the simulation
    t0 = time.time()
    t, x_com, x_left, x_right, p_left, p_right, D = mdrom.RK3_rollout(x0_com, p_left, p_right, U, D0)
    tf = time.time()
    print('Simulation time: ', tf - t0)

    # print(t.shape)
    # print(x_com.shape)
    # print(x_left.shape)
    # print(x_right.shape)
    # print(p_left.shape)
    # print(p_right.shape)
    # print(len(D))

    # save the data into CSV files
    np.savetxt('./data/double/time.csv', t, delimiter=',')
    np.savetxt('./data/double/state_com.csv', x_com.T, delimiter=',')
    np.savetxt('./data/double/state_left.csv', x_left.T, delimiter=',')
    np.savetxt('./data/double/state_right.csv', x_right.T, delimiter=',')
    np.savetxt('./data/double/pos_left.csv', p_left.T, delimiter=',')
    np.savetxt('./data/double/pos_right.csv', p_right.T, delimiter=',')
    np.savetxt('./data/double/domain.csv', D, delimiter=',', fmt='%s')

