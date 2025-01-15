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
    N: int         # prediction horizon
    dt: float      # time step [s]
    K: int         # number of rollouts
    Nu: int        # number of input knots
    interp: str    # interpolation method, 'Z' for zero order hold, 'L' for linear
    Q: np.array    # integrated state cost matrix
    Qf: np.array   # terminal state cost matrix
    l0_rate_penalty: float  # control input penalty on rate of change
    theta_rate_penalty: float  # control input penalty on rate of change
    N_elite: int   # number of elite rollouts
    CEM_iters: int # number of CEM iterations

@dataclass
class ParametricDistribution:
    """
    Parametric distribution parameters
    """
    family: str          # distribution family, 'G' Gaussian, 'U' Uniform
    mean: np.array       # mean of the distribution
    cov: np.array        # covaraince of the distribution (Guassian only)
    diag_only: bool      # diagonal only covariance (Gaussian only)
    min_var: np.array    # minimum variance of the distribution (Gaussian only)
    lb: np.array         # lower bound of the distribution (Uniform only)
    ub: np.array         # upper bound of the distribution (Uniform only)

###################################################################################################
# DYNAMICS
###################################################################################################

class MDROM:
    """
     Class for the Planar Spring loaded inverted pendulum (SLIP) model
    """
    def __init__(self, 
                 system_params, 
                 control_params):
        
        # initialize system parameters
        self.m = system_params.m
        self.g = system_params.g
        self.k = system_params.k
        self.b = system_params.b
        self.l0 = system_params.l0
        self.r_min = system_params.r_min
        self.r_max = system_params.r_max
        self.theta_min = system_params.theta_min
        self.theta_max = system_params.theta_max
        self.rdot_lim = system_params.rdot_lim
        self.thetadot_lim = system_params.thetadot_lim
        self.torque_ankle = system_params.torque_ankle
        self.torque_ankle_lim = system_params.torque_ankle_lim
        self.torque_ankle_kp = system_params.torque_ankle_kp
        self.torque_ankle_kd = system_params.torque_ankle_kd

        # initialize control parameters
        self.dt = control_params.dt
        self.N = control_params.N
        self.interp = control_params.interp

    #################################  DYNAMICS  #################################

    def dynamics(self, x_sys, p_foot, u, d):
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
        p_com = np.array([[x_sys[0]],  # position of the center of mass
                          [x_sys[1]]]).reshape(2, 1)
        v_com = np.array([[x_sys[2]],  # velocity of the center of mass
                          [x_sys[3]]]).reshape(2, 1)

        # Flight domain (F)
        if d == 'F':

            # no torque applied
            torque_ankle = 0.0

            # compute the dynamics
            a_com = np.array([[0],  
                              [-g]])
            v_leg = np.array([[u[0]],
                              [u[1]]])

            xdot = np.vstack((v_com, a_com, v_leg))

        # Ground stance domain (G)
        elif d == 'G':

            # unpack leg positions
            l0_hat = x_sys[4]       # command leg length

            # compute the leg state
            r_vec = p_foot - p_com
            r_norm = np.linalg.norm(r_vec)
            r_hat = r_vec / r_norm

            # compute passive leg angle
            rdot_vec = -v_com
            rdot_x = rdot_vec[0][0]
            rdot_z = rdot_vec[1][0]
            thetadot = r_hat.T @ np.array([[-rdot_z], [rdot_x]]) / r_norm # TODO: isnt it divided by r, See my math to confirm... I think it is? Changed it

            # compute the groundreaction force
            lambd = -r_hat * (k * (l0_hat - r_norm) + b * (v_com.T @ r_hat))

            # compute the equivalent force from ankle torque (force prependicular to the leg applied at COM)
            if self.torque_ankle == True:
                # references to track
                theta_command = x_sys[5][0] # theta_hat
                thetadot_command = u[1]  # thetadot_hat

                # get current leg state
                r_x = r_vec[0][0]
                r_z = r_vec[1][0]
                theta = -np.arctan2(r_x, -r_z)

                # compute the ankle torque
                kp = self.torque_ankle_kp
                kd = self.torque_ankle_kd
                torque_ankle = kp * (theta_command - theta) + kd * (thetadot_command - thetadot[0][0])
                
                # saturate the torque
                torque_ankle = np.clip(torque_ankle, -self.torque_ankle_lim, self.torque_ankle_lim)

                f_unit = np.array([[np.cos(theta)], 
                                   [-np.sin(theta)]])
                f_mag = torque_ankle / r_norm
                f_com = f_mag * f_unit

            # no leg torque, set the force to zero
            elif self.torque_ankle == False:
                torque_ankle = 0.0
                f_com = np.array([[0], [0]])

            # compute the dynamics
            a_com = (1/m) * lambd + np.array([[0], [-g]]) + (1/m) * f_com
            v_leg = np.array([[u[0]],
                              [u[1]]])  

            xdot = np.vstack((v_com, a_com, v_leg))

        return xdot, torque_ankle

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
    def RK3_rollout(self, Tx, Tu, x0_sys, p0_foot, U, D0):
        """
        Runge-Kutta 3rd order integration scheme
        """
        # integration parameters
        dt = self.dt
        N = self.N

        # make the containers
        xt_sys = np.zeros((6, N))
        xt_leg = np.zeros((4, N))
        xt_foot = np.zeros((4, N))
        ut = np.zeros((2, N))
        ut_ankle = np.zeros((1, N))
        lambd_t = np.zeros((2, N))
        D = [None for _ in range(N)]

        # compute leg state
        u0 = self.interpolate_control_input(0, Tu, U)
        x0_leg, lambd = self.update_leg_state(x0_sys, p0_foot, u0, D0)
        x0_foot = self.update_foot_state(x0_sys, x0_leg, p0_foot, D0)

        # initial conditions
        xt_sys[:, 0] = x0_sys.reshape(6) 
        xt_leg[:, 0] = x0_leg.reshape(4)
        xt_foot[:, 0] = x0_foot.reshape(4)
        ut[:, 0] = u0.reshape(2)
        lambd_t[:, 0] = lambd.reshape(2)  
        D[0] = D0

        # RK3 integration
        xk_sys = x0_sys
        xk_leg = x0_leg
        xk_foot = x0_foot
        p_foot = p0_foot
        Dk = D0
        contacts = self.contact_identification(D0)
        viability = True
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
            f1, _ = self.dynamics(xk_sys, 
                                  p_foot,
                                  u1, 
                                  Dk)
            f2, tau_ankle = self.dynamics(xk_sys + dt/2 * f1, 
                                  p_foot, 
                                  u2, 
                                  Dk)
            f3, _ = self.dynamics(xk_sys - dt*f1 + 2*dt*f2, 
                                  p_foot,
                                  u3, 
                                  Dk)

            # RK3 step and update all states
            xk_sys = xk_sys + (dt/6) * (f1 + 4*f2 + f3)
            xk_leg, lambd = self.update_leg_state(xk_sys, p_foot, u3, Dk)
            xk_foot = self.update_foot_state(xk_sys, xk_leg, p_foot, Dk)

            # check if hit switching surfaces and update state if needed
            checked_contacts = self.check_switching(xk_sys, xk_leg, xk_foot, contacts)

            # if a change in contact detected
            if checked_contacts != contacts:
                
                # check if the take off state is viable
                # TODO: should I add viability w.r.t. to friction cone?
                if (contacts==True) and (checked_contacts==False):
                    # check viability condition
                    viability = self.in_viability_kernel(xk_sys, xk_leg)
                    
                    # you're going to die, exit the for loop
                    if viability == False:
                        # print('Exited Viability Kernel, terminating rollout...')
                        break

                # update all states
                xk_sys, xk_leg, xk_foot, lambd, p_foot = self.reset_map(xk_sys, 
                                                                        xk_leg, 
                                                                        xk_foot, 
                                                                        p_foot, 
                                                                        u3,
                                                                        contacts, 
                                                                        checked_contacts)

                # update domain and contact info
                Dk = self.domain_identification(checked_contacts)
                contacts = checked_contacts

            # store the states
            xt_sys[:, i+1] = xk_sys.reshape(6)
            xt_leg[:, i+1] = xk_leg.reshape(4)
            xt_foot[:, i+1] = xk_foot.reshape(4)
            ut[:, i+1] = u3.reshape(2)
            ut_ankle[:, i+1] = tau_ankle
            lambd_t[:, i+1] = lambd.reshape(2)

            # store the domain
            D[i+1] = Dk

        # package into a solution
        sol = (Tx, xt_sys, xt_leg, xt_foot, ut, ut_ankle, lambd_t, D, viability)

        return sol

    #################################  LEG STATE UPDATES  #################################

    # update leg polar state
    def update_leg_state(self, x_sys, p_foot, u, D):
        """
        Update the leg state based on the COM state and control input
        """
        # in flight (state governed by input)
        if D == 'F':
            # build the state
            r = x_sys[4][0]
            theta = x_sys[5][0]
            rdot = u[0]
            thetadot = u[1]

            # compute leg force
            lambd = np.array([[0], [0]])

            # pack into state vectors
            x_leg = np.array([[r],
                              [theta],
                              [rdot],
                              [thetadot]])
        
        # in leg support (leg state governed by dynamics)
        if D == 'G':
            # unpack COM state
            p_com = np.array([[x_sys[0]],  
                              [x_sys[1]]]).reshape(2, 1)
            v_com = np.array([[x_sys[2]],
                              [x_sys[3]]]).reshape(2, 1)

            # compute relevant vectors
            r_vec = p_foot - p_com
            r_hat = r_vec / np.linalg.norm(r_vec)
            rdot_vec = -v_com

            r_x = r_vec[0]
            r_z = r_vec[1]
            rdot_x = rdot_vec[0]
            rdot_z = rdot_vec[1]

            # compute the leg state in polar coordinates
            r = np.linalg.norm(r_vec)
            rdot = -v_com.T @ r_hat
            theta = -np.arctan2(r_x, -r_z)
            thetadot = (r_z * rdot_x - r_x * rdot_z) / r**2  # TODO: isnt it divided by r^2, See my math to confirm... I think it is? Changed it

            # compute leg force
            l0_hat = x_sys[4][0]
            lambd = -r_hat * (self.k * (l0_hat - r) - self.b * rdot)

            # pack into state vectors
            x_leg = np.array([[r],
                              [theta[0]],
                              [rdot[0][0]],
                              [thetadot[0]]])

        return x_leg, lambd

    # compute foot position and veclocity in world
    def update_foot_state(self, x_sys, x_leg, p_foot, D):
        """
        Get the foot state in the world frame. Usefull for checking switching surfaces
        and recording history of foot locations
        """

        # pin the foot
        if D == 'G':

            px_foot = p_foot[0][0]
            pz_foot = p_foot[1][0]
            vx_foot = 0.0
            vz_foot = 0.0

        # in swing
        elif D == 'F':

            # get the com state
            px_com = x_sys[0][0]
            pz_com = x_sys[1][0]
            vx_com = x_sys[2][0]
            vz_com = x_sys[3][0]

            # get the leg state
            r = x_leg[0][0]
            theta = x_leg[1][0]
            rdot = x_leg[2][0]
            thetadot = x_leg[3][0]

            # foot position in world frame frame 
            px_foot = px_com - r * np.sin(theta)
            pz_foot = pz_com - r * np.cos(theta)
            vx_foot = vx_com - rdot * np.sin(theta) - r * thetadot * np.cos(theta)
            vz_foot = vz_com - rdot * np.cos(theta) + r * thetadot * np.sin(theta)

        # update the foot state
        x_foot = np.array([[px_foot],       
                           [pz_foot],
                           [vx_foot],
                           [vz_foot]])
   
        return x_foot

    #################################  SWITCHING  #################################

    # check switching surfaces
    def check_switching(self, x_sys, x_leg, x_foot, contact):
        """
        Check if the state has hit any switching surfaces
        """
         # In Flight (F)
        if contact == False:
            # check for touchdown
            contact_result = self.S_TD(x_foot)
        # In Ground(G)
        elif contact == True:
            # check for takeoff
            contact_result = self.S_TO(x_sys, x_leg)

        return contact_result

    # Touch-Down (TD) Switching Surface -- checks individual legs
    def S_TD(self, x_foot):
        """
        Check if a leg has touched the ground
        """
        # get relevant states
        pz_foot = x_foot[1][0]
        vz_foot = x_foot[3][0]

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
    def S_TO(self, x_sys, x_leg):
        """
        Check if a leg has taken off from the ground
        """
        # get relevant states
        r = x_leg[0]
        rdot = x_leg[2]
        l0_hat = x_sys[4][0]

        # check the switching surface conditions
        # nom_length = (r >= self.l0)      # the leg is at its nominal uncompressed length
        nom_length = (r >= l0_hat)         # the leg is at its nominal uncompressed length
        pos_vel = (rdot >= 0.0)            # the leg is going in uncompressing direction
        takeoff = nom_length and pos_vel   # if true, leg has taken off into flight

        # return the contact result
        if takeoff==True:
            contact = False
        elif takeoff==False:
            contact = True

        return contact
    
    # check if given the take off state is viable, is there an input to save you from eating dirt
    def in_viability_kernel(self, x_com_TO, x_leg_TO):
        """
        Check if the take off state is viable. 
        Given the take off state is there an input that can save you?
        """
        # compute the apex time if any 
        pz_com_TO = x_com_TO[1]
        vz_com_TO = x_com_TO[3]

        # compute the apex height
        if vz_com_TO <= 0.0:
            pz_com_max = pz_com_TO
        else:
            t_apex = vz_com_TO / self.g
            pz_com_max = pz_com_TO + vz_com_TO * t_apex - 0.5 * self.g * t_apex**2

        # if you can't retract your leg enough by the time you are at the apex, 
        # there is no control input that can save you, you are going to eat dirt
        # this is only becuase we have infinite swing velocity, TODO: eventually will have swing vel too.
        if self.r_min >= pz_com_max:
            viable = False
        else:  
            viable = True   

        return viable

    # apply a reset map
    def reset_map(self, x_sys, x_leg, x_foot, p_foot, u, contact_prev, contact_new):
        """
        Changing the foot location effectively applies a reset map to the system.
        """
        # Flight to Ground
        if (contact_prev == False) and (contact_new == True):
            # update the ground foot location (based on heuristic)
            px_foot = x_foot[0][0]
            # pz_foot = x_foot[1][0]
            # vx_foot = x_foot[2][0]
            # vz_foot = x_foot[3][0]
            # px_foot_post = px_foot - (vx_foot/vz_foot) * pz_foot # there's different approximations of this
            px_foot_post = px_foot
            pz_foot_post = 0.0
            p_foot_post = np.array([[px_foot_post],
                                    [pz_foot_post]])

            # update the leg
            x_leg_post, lambd_post = self.update_leg_state(x_sys, p_foot_post, u, 'G')

            # update the system
            x_sys_post = x_sys
            x_sys_post[4] = x_leg_post[0]
            x_sys_post[5] = x_leg_post[1]

            # update the foot state
            x_foot_post = self.update_foot_state(x_sys_post, x_leg_post, p_foot_post, 'G')

        # Ground to Flight
        elif (contact_prev == True) and (contact_new == False):
            # update the ground foot location (flight phase, no contact)
            p_foot_post = np.array([[None],
                                    [None]])
            
            # update the leg
            x_leg_post= x_leg
            x_leg_post[2] = u[0]
            x_leg_post[3] = u[1]
            lambd_post = np.array([[0], [0]])

            # update the system  # TODO: Shouldnt I just keep integrating the command?
            x_sys_post = x_sys
            x_sys_post[4] = x_leg_post[0]  # the l0 & theta_hat, commands were being integrating during
            x_sys_post[5] = x_leg_post[1]  # ground phase, now snap back to the actual leg state
            
            # update the foot state
            x_foot_post = self.update_foot_state(x_sys, x_leg, p_foot_post, 'F')

        return x_sys_post, x_leg_post, x_foot_post, lambd_post, p_foot_post

    #################################  DOMAIN/CONTACT MAPPING  #################################
    
    # Domain mapping function
    def domain_identification(self, contact):
        """
        Map the contact state to a unique domain
        """
        # Flight (F)
        if contact == False:
            D = 'F'
        # Double leg support (G)
        elif contact == True:
            D = 'G'
        
        return D
    
    # Domain mapping function
    def contact_identification(self, D):
        """
        Simple contact identification given a domain
        """
        # Flight (F)
        if D == 'F':
            contact = False
        # Ground Support (G)
        elif D == 'G':
            contact = True
        
        return contact

###################################################################################################
# SAMPLING PREDICTIVE CONTROL
###################################################################################################

class PredictiveController:
    """
    A class that handles all of the control via sample predictive
    control. 
    """
    def __init__(self, mdrom, sys_params, ctrl_params, distr_params):
        
        # make internal objects
        self.mdrom = mdrom
        self.sys_params = sys_params
        self.ctrl_params = ctrl_params
        self.distr_params = distr_params

        # predictive control parameters
        self.dt = ctrl_params.dt
        self.N = ctrl_params.N
        self.K = ctrl_params.K
        self.Nu = ctrl_params.Nu
        self.N_elite = ctrl_params.N_elite

        # distribution parameters
        self.mean = distr_params.mean
        self.cov = distr_params.cov
        self.lb = distr_params.lb
        self.ub = distr_params.ub

    # sample an input trajectory given a distribution
    def sample_input_trajectory(self):
        """
        Given a distribution and number of control points, sample an input trajectory
        """
        # get the distribution parameters
        mean = self.mean

        # sample from Gaussian dsitribution
        if self.distr_params.family == 'G':
            cov = self.cov
        
            # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
            L = np.linalg.cholesky(cov)
            Uhat_vec = np.random.randn(mean.shape[0])
            U_vec = mean.flatten() + L @ Uhat_vec
            # U_vec = np.random.multivariate_normal(mean.flatten(), cov).T # (SLOW!! DO NOT USE)

        # sample from uniform distirbution
        elif self.distr_params.family == 'U':
            lb = self.lb
            ub = self.ub
            U_vec = np.random.uniform(lb.flatten(), ub.flatten()).T

        # unflatten the trajectory
        U_traj = np.zeros((2, self.Nu))
        for i in range(0, self.Nu):
            U_traj[:, i] = U_vec[ 2*i : 2*i+2 ]

        # saturate the inputs
        U_traj[0, :] = np.clip(U_traj[0, :], -self.sys_params.rdot_lim, self.sys_params.rdot_lim)
        U_traj[1, :] = np.clip(U_traj[1, :], -self.sys_params.thetadot_lim, self.sys_params.thetadot_lim)

        return U_traj

    # generate slip integrator dynamics solution
    def generate_reference_trajectory(self, x0_com, pz_des, vx_des):
        """
        Based on where the COM currently is, generate a reference trajectory
        with constant velocity and desired height
        """
        # some MPC parameters
        dt = self.ctrl_params.dt
        N = self.ctrl_params.N

        # initial position 
        px_0 = x0_com[0]

        # initilize the reference trajectory
        X_des = np.zeros((8, N))
        X_des[1, :] = np.ones((1, N)) * pz_des
        X_des[2, :] = np.ones((1, N)) * vx_des
        X_des[4, :] = np.ones((1, N)) * self.sys_params.l0

        # generate the reference trajectory
        for i in range(0, N):
            t = i * dt
            px_des = px_0 + vx_des * t
            X_des[0, i] = px_des[0]

        return X_des

    # cost function evaluation
    def cost_function(self, X_des, S, U):
        """
        Evaulate the cost function given a propagated and desired trajecotry.
        """
        # unpack solution
        # sol = (T, xt_sys, xt_leg, xt_foot, lambd_t, D, viability)
        # T = S[0]
        X_sys = S[1]
        X_leg = S[2]
        # X_foot = S[3]
        # U = S[4]
        # U_ankle = S[5]
        # Lambd = S[6]
        # D = S[7]
        # viability = S[8]

        # full SLIP state, com and leg states
        X_com = X_sys[:4, :]
        X = np.vstack((X_com, X_leg))

        # cost function variables
        Q = self.ctrl_params.Q
        Qf = self.ctrl_params.Qf

        # input rate of change cost (discourage rapid changes in input)
        l0_rate_penalty = self.ctrl_params.l0_rate_penalty
        theta_hat_rate_penalty = self.ctrl_params.theta_rate_penalty
        U_l0 = U[0, :]
        U_theta_hat = U[1, :]
        U_l0_rate = np.diff(U_l0)
        U_theta_hat_rate = np.diff(U_theta_hat)
        input_rate_cost = l0_rate_penalty * np.linalg.norm(U_l0_rate) + theta_hat_rate_penalty * np.linalg.norm(U_theta_hat_rate)

        # stage cost
        state_cost = 0.0
        for i in range(0, self.N-1):
            # compute error
            xi = X[:, i]   
            xi_des = X_des[:, i]
            e_com = (xi - xi_des).reshape(8, 1)

            # compute cost
            cost = e_com.T @ Q @ e_com
            state_cost += cost

        # terminal cost
        xf= X[:, self.N-1]
        xf_des = X_des[:, self.N-1]
        ef = (xf - xf_des).reshape(8, 1)
        terminal_cost = ef.T @ Qf @ ef

        # total cost
        total_cost = state_cost + terminal_cost + input_rate_cost
        total_cost = total_cost[0][0]

        return total_cost
    
    # perform horizon rollouts
    def monte_carlo(self, x0_sys, p0_foot, D0, U_list, pz_des, vx_des):

        # generate the time arrays
        Tx = np.arange(0, self.N) * self.dt
        Tu = np.linspace(0, Tx[-1], self.Nu)

        # preallocate the the containers
        J_list = np.zeros((1, self.K))            # cost function list
        S_list = [None for _ in range(self.K)]    # solution list

        # generate the reference trajectory
        X_des = self.generate_reference_trajectory(x0_sys, pz_des, vx_des)

        # perform the rollouts
        for k in range(0, self.K):

            # sample an input trajectory
            U = U_list[:, :, k]
            
            # rollout the dynamics under the input trajectory
            S = self.mdrom.RK3_rollout(Tx, Tu, x0_sys, p0_foot, U, D0)

            # evaluate the cost function
            J = self.cost_function(X_des, S, U)

            # save the data
            J_list[:,k] = J
            S_list[k] = S

        # reorder the rollouts based on cost
        idx = np.argsort(J_list)
        idx = idx[0, :]
        J_list = J_list[:, idx]
        U_list = U_list[:, :, idx]
        S_list = [S_list[i] for i in idx]

        return S_list, U_list, J_list
    
    # perform sampling based predictive control
    def predictive_control(self, x0_sys, p_foot, D0, pz_des, vx_des):
        """
        Perform sampling predictive control
        """
        # perform cross entropy iterations
        for i in range(0, self.ctrl_params.CEM_iters):

            print('-'*50)
            print('CEM Iteration: ', i+1)
            print('-'*50)

            # generate list of input trajecotries
            U_list = np.zeros((2, self.Nu, self.K))
            for i in range(0, self.K):
                U_list[:, :, i] = self.sample_input_trajectory()

            # perform the monte carlo simulation
            t0 = time.time()
            S_ascending, U_ascending, J_ascending = ctrl.monte_carlo(x0_sys, p_foot, D0, U_list, pz_des, vx_des)
            tf = time.time()
            print('Elapsed time: ', tf - t0)
            print('Average time spent per rollout: ', (tf - t0) / control_params.K) 
            print('Percent Failure Rate: ', np.sum(J_ascending==np.inf) / control_params.K)
            print('Failure: %d/%d' % (np.sum(J_ascending==np.inf), control_params.K))
            print('Norm Covariance: ', np.linalg.norm(self.cov)) 

            # compute the new distribution
            if self.distr_params.family == 'G':
                self.mu, self.cov = self.update_gaussian_distribution(U_ascending)
                print('updated gaussian')
            elif self.distr_params.family == 'U':
                self.lb, self.ub = self.update_uniform_distribution(U_ascending)
                print('updated uniform')

        print('Finished preditcion horizon optimization')

        return S_ascending[0]

    # update the Gaussian distribution
    def update_gaussian_distribution(self, U_list):
        """
        Average the control inputs of the elite rollouts
        """        
        # select the elite rollouts
        U_elite = U_list[:, :, 0:self.N_elite]

        # build a input data matrix
        U_data = np.zeros((2 * self.Nu, self.N_elite)) # each column is a vectorized input signal
        for i in range(0, self.N_elite):
            
            # container for the input signal
            U_t = U_elite[:,:,i]
            U_t_flat = np.zeros((2 * self.Nu, 1))
            
            # insert into the input data matrix
            for j in range(0, self.Nu):
                U_t_flat[2*j:2*j+2] = U_t[:, j].reshape(2, 1)
            U_data[:, i] = U_t_flat.flatten()

        # compute the mean and covariance
        mu = np.mean(U_data, axis=1).reshape(2 * self.Nu, 1)
        cov = (1/(self.N_elite-1)) * (U_data - mu) @ (U_data - mu).T
        cov = np.diag(np.diag(cov))

        # TODO: add a minimum variance constraint

        return mu, cov
    
    # update the Uniform distribution
    def update_uniform_distribution(self, U_list):
        """
        Average the control inputs of the elite rollouts
        """        
        lb = 0
        ub = 0

        return lb, ub

#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":

    # SYSTEM PARAMS
    system_params = SystemParams(m=5.0, 
                                 g=9.81, 
                                 k=5000.0, 
                                 b=75.0,
                                 l0=0.65,
                                 r_min=0.4,
                                 r_max=0.8,
                                 theta_min=-np.pi/3,
                                 theta_max=np.pi/3,
                                 rdot_lim=1.0,
                                 thetadot_lim=np.pi,
                                 torque_ankle=True,
                                 torque_ankle_lim=5,
                                 torque_ankle_kp=125,
                                 torque_ankle_kd=10)

    # CONTROl PARAMS
    Q_diags = np.array([0.0, 5.0, 1.0, 0.05,     # COM: px, pz, vx, vz 
                        0.5, 0., 0.05, 0.01])   # LEG: r, theta, rdot, thetadot
    Qf_diags = 5 * Q_diags
    Q = np.diag(Q_diags)
    Qf = np.diag(Qf_diags)
    l0_rate_penalty = 0.01
    theta_rate_penalty = 0.01
    control_params = PredictiveControlParams(N=100, 
                                             dt=0.01, 
                                             K=250,
                                             Nu=25,
                                             interp='L',
                                             Q=Q,
                                             Qf=Qf,
                                             l0_rate_penalty=l0_rate_penalty,
                                             theta_rate_penalty=theta_rate_penalty,
                                             N_elite=25,
                                             CEM_iters=10)

    # create parametric distribution parameters
    mean_r = 0.0             # [m/s]
    mean_theta = 0.0         # [rad/s]
    std_dev_r = 1.0          # [m/s]
    std_dev_theta = np.pi  # [rad/s]

    mean = np.array([[mean_r],              # r [m]
                     [mean_theta]])         # theta [rad]
    
    # Gaussian distribution
    std_dev = np.array([std_dev_r**2,       # r [m]
                        std_dev_theta**2])  # theta [rad]
    mean_initial = np.tile(mean, (control_params.Nu, 1))
    std_dev_matrix = np.diag(std_dev)
    I = np.eye(control_params.Nu)
    cov_initial = np.kron(I, std_dev_matrix)

    # Uniform distribution
    lb_rdot = -0.75    # [m/s]
    ub_rdot = 0.75     # [m/s]
    lb_thetadot = -1.0 # [rad/s]
    ub_thetadot = 1.0  # [rad/s]

    lb = np.array([[lb_rdot],      # r [m]
                   [lb_thetadot]]) # theta [rad]
    ub = np.array([[ub_rdot],      # r [m]
                   [ub_thetadot]]) # theta [rad]
    ones_vec = np.ones((control_params.Nu, 1))
    lb_initial = np.kron(ones_vec, lb)
    ub_initial = np.kron(ones_vec, ub)

    # DISTRIBUTION PARAMS    
    distribution_params = ParametricDistribution(family='G',
                                                 mean=mean_initial,
                                                 cov=cov_initial,
                                                 diag_only=True,
                                                 min_var = np.array([[0.1**2], [np.pi/5**2]]),
                                                 lb=lb_initial,
                                                 ub=ub_initial)

    # declare reduced order model object
    mdrom = MDROM(system_params, control_params)

    # create a predictive controller object
    ctrl = PredictiveController(mdrom, system_params, control_params, distribution_params)

    # initial conditions
    x0_sys = np.array([[0.0],              # px com
                       [0.7],              # pz com
                       [1.0],                # vx com
                       [1.0],                # vz com
                       [system_params.l0], # l0 command
                       [0.0]])             # theta command
    p0_foot = np.array([[0.0], [0.0]])
    D0 = 'F'  # initial domain

    # desired velocity and height
    pz_des = 0.60
    vx_des = 0.0

    # Prediciton horizon optimization
    sol = ctrl.predictive_control(x0_sys, p0_foot, D0, pz_des, vx_des)

    t = sol[0]
    x_sys = sol[1]
    x_leg = sol[2]
    x_foot = sol[3]
    u = sol[4]
    u_ankle = sol[5]
    lambd = sol[6]
    d = sol[7]
    viability = sol[8]

    print(t.shape)
    print(x_sys.shape)
    print(x_leg.shape)
    print(x_foot.shape)
    print(u.shape)
    print(u_ankle.shape)
    print(lambd.shape)
    print(len(d))
    print(viability)

    np.savetxt('./data/slip/time.csv', t, delimiter=',')
    np.savetxt('./data/slip/state_com.csv', x_sys.T, delimiter=',')
    np.savetxt('./data/slip/state_leg.csv', x_leg.T, delimiter=',')
    np.savetxt('./data/slip/state_foot.csv', x_foot.T, delimiter=',')
    np.savetxt('./data/slip/input.csv', u.T, delimiter=',')
    np.savetxt('./data/slip/input_ankle.csv', u_ankle.T, delimiter=',')
    np.savetxt('./data/slip/lambd.csv', lambd.T, delimiter=',')
    np.savetxt('./data/slip/domain.csv', d, delimiter=',', fmt='%s')
