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
    m: float            # mass [kg]
    g: float            # gravity [m/s^2]
    k: float            # spring stiffness [N/m]
    b: float            # damping coefficient [Ns/m]
    l0: float           # spring free length [m]
    r_min: float        # leg min length [m]
    r_max: float        # leg max length [m]
    theta_min: float    # revolute leg min angle [rad]
    theta_max: float    # revolute leg max angle [rad]
    rdot_lim: float     # leg vel max velocity [m/s]
    thetadot_lim: float # revolute leg max angular velocity [rad/s]

@dataclass
class PredictiveControlParams:
    """
    Predictive control parameters
    """
    N: int       # prediction horizon
    dt: float    # time step [s]
    K: int       # number of rollouts
    interp: str  # interpolation method, 'Z' for zero order hold, 'L' for linear
    Q: np.array   # integrated state cost matrix
    Qf: np.array  # terminal state cost matrix

@dataclass
class ParametricDistribution:
    """
    Parametric distribution parameters
    """
    family: str          # distribution family, 'G' Gaussian, 'U' Uniform
    mean: np.array       # mean of the distribution
    cov: np.array        # covaraince of the distribution (Guassian only)
    lb: np.array         # lower bound of the distribution (Uniform only)
    ub: np.array         # upper bound of the distribution (Uniform only)

#######################################################################
# DYNAMICS
#######################################################################

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
            # compute the dynamics
            a_com = np.array([[0],  
                              [-g]])
            v_leg = np.array([[u[0]],
                              [u[1]]])
            xdot = np.vstack((v_com, a_com, v_leg))

        # Ground stance domain (G)
        elif d == 'G':

            # unpack leg positions
            l0_hat = x_sys[4]     # command leg length

            # compute the leg state
            r_vec = p_foot - p_com
            r_norm = np.linalg.norm(r_vec)
            r_hat = r_vec / r_norm

            # compute passive leg angle
            rdot_vec = -v_com
            rdot_x = rdot_vec[0]
            rdot_z = rdot_vec[1]
            thetadot_hat = r_hat.T @ np.array([[rdot_x], [-rdot_z]])

            # compute the groundreaction force
            lambd = -(r_hat/m) * (k * (l0_hat - r_norm) + b * (v_com.T @ r_hat))

            # compute the dynamics
            a_com = lambd + np.array([[0], [-g]])
            v_leg = np.array([[u[0]],
                              [thetadot_hat]])
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
    def RK3_rollout(self, x0_sys, p0_foot, U, D0):
        """
        Runge-Kutta 3rd order integration scheme
        """
        # integration parameters
        dt = self.dt
        N = self.N

        # make the containers
        Tx = np.arange(0, N) * dt
        Tu = np.arange(0, N-1) * dt
        xt_sys = np.zeros((6, N))
        xt_leg = np.zeros((4, N))
        pt_foot = np.zeros((2, N))
        D = [None for _ in range(N)]

        # compute leg state

        # initial conditions
        xt_sys[:, 0] = x0_sys.reshape(6) 
        pt_foot[:, 0] = p0_foot.reshape(2)
        D[0] = D0

        # RK3 integration
        xk_sys = x0_sys
        # xk_leg = x0_leg
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
            f1 = self.dynamics(xk_sys, 
                               p_foot,
                               u1, 
                               Dk)
            f2 = self.dynamics(xk_sys + dt/2 * f1, 
                               p_foot, 
                               u2, 
                               Dk)
            f3 = self.dynamics(xk_sys - dt*f1 + 2*dt*f2, 
                               p_foot,
                               u3, 
                               Dk)

            # RK3 step and update all states
            xk_sys = xk_sys + (dt/6) * (f1 + 4*f2 + f3)
            xk_leg = self.update_leg_state(xk_sys, p_foot, u1, Dk)
            print(xk_sys)
            print(xk_leg)
            exit(0)

            # check if hit switching surfaces and update state if needed
            checked_contacts = self.check_switching(xk_sys, xk_leg, u1, contacts)
            
            # if a change in contact detected
            if checked_contacts != contacts:
                
                # check if the take off state is viable
                # TODO: should I check viability of the touch down?
                if (contacts==True) and (checked_contacts==False):
                    viability = self.in_viability_kernel(xk_sys, xk_leg)
                    
                    # you're going to die, exit the for loop
                    if viability == False:
                        print('Exited the viability kernel, exited rollout...')
                        break

                # update the feet positions
                p_foot = self.apply_reset(xk_sys, xk_leg, checked_contacts)
                
                # update domain and contact info
                Dk = self.domain_identification(checked_contacts)
                contacts = checked_contacts

            # store the states
            xt_sys[:, i+1] = xt_sys.reshape(4)
            xt_leg[:, i+1] = xk_leg.reshape(4)
            
            # store the foot positions
            if Dk == 'F':
                pt_foot[:, i+1] = self.get_foot_state_in_world(xt_sys, xk_leg, pz_zero=False).flatten()
            elif Dk == 'G':
                pt_foot[:, i+1] = p_foot.reshape(2)

            # store the domain
            D[i+1] = Dk

        # package into a solution
        sol = (Tx, xt_sys, xt_leg, pt_foot, D, viability)

        return sol
    
    #################################  STATE UPDATES  #################################

    # update leg polar state
    def update_leg_state(self, x_sys, p_foot, u, D):
        """
        Update the leg state based on the COM state and control input
        """
        # unpack COM state
        p_com = np.array([[x_sys[0]],  
                          [x_sys[1]]]).reshape(2, 1)
        v_com = np.array([[x_sys[2]],
                          [x_sys[3]]]).reshape(2, 1)

        # in flight (state governed by input)
        if D == 'F':
            # build the state
            r = x_sys[4][0]
            theta = x_sys[5][0]
            rdot = u[0]
            thetadot = u[1]

            print(r, theta, rdot, thetadot)

            # pack into state vectors
            x_leg = np.array([[r],
                              [theta],
                              [rdot],
                              [thetadot]])
        
        # in leg support (leg state governed by dynamics)
        if D == 'G':

            # compute relevant vectors
            r_vec = p_foot - p_com
            r_hat = r_vec / np.linalg.norm(r_vec)
            rdot_vec = -v_com

            r_x = r_vec[0]
            r_z = r_vec[1]
            rdot_x = rdot_vec[0]
            rdot_z = rdot_vec[1]

            # compute the left leg state in polar coordinates
            r = np.linalg.norm(r_vec)
            rdot = -v_com.T @ r_hat
            theta = -np.arctan2(r_x, -r_z)
            thetadot = (r_z * rdot_x - r_x * rdot_z) / r

            # pack into state vectors
            x_leg = np.array([[r],
                              [theta[0]],
                              [rdot[0][0]],
                              [thetadot[0]]])

        return x_leg

    # compute foot position and veclocity in world
    def get_foot_state_in_world_in_world(self, x_sys, x_leg, pz_zero):
        """
        Get the foot state in the world frame. Usefull for checking switching surfaces
        and recording history of foot locations
        """
        # unpack the states
        p_com = np.array([[x_sys[0]],
                          [x_sys[1]]]).reshape(2, 1)
        
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
    def check_switching(self, x_sys, x_leg, u, contact):
        """
        Check if the state has hit any switching surfaces
        """
         # In Flight (F)
        if contact == False:
            # check for touchdown
            contact_result = self.S_TD(x_sys, x_leg)
        # In Ground(G)
        elif contact == True:
            # check for takeoff
            u_leg = u[0]
            contact_result = self.S_TO(x_sys, x_leg, u_leg)

        return contact_result

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
    # TODO: consoder switching only when the leg is zero force
    def S_TO(self, x_com, x_leg, u_leg):
        """
        Check if a leg has taken off from the ground
        """
        # get relevant states
        r = x_leg[0]
        rdot = x_leg[2]

        # check the switching surface conditions
        # set by input
        nom_length = (r >= u_leg)        # the leg is at its nominal uncompressed length
        # nom_length = (r >= self.l0)      # the leg is at its nominal uncompressed length
        pos_vel = (rdot >= 0.0)          # the leg is going in uncompressing direction
        takeoff = nom_length and pos_vel # if true, leg has taken off into flight

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
    def apply_reset(self, x_com, x_leg, contact):
        """
        Changing the foot location effectively applies a reset map to the system.
        """
        # TODO: eventually you will need to have a swing leg velocity as well

        # Left leg update
        if contact == True:
            p_foot = self.get_foot_state_in_world(x_com, x_leg, pz_zero=True)
        elif contact == False:
            p_foot = np.array([[None],
                               [None]])

        return p_foot

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

#######################################################################
# SAMPLING PREDICTIVE CONTROL
#######################################################################

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

    # sample an input trajectory given a distribution
    def sample_input_trajectory(self):
        """
        Given a distribution, sample an input trajectory
        """
        # get the distribution parameters
        mean = self.distr_params.mean
        
        # sample from Gaussian dsitribution
        if self.distr_params.family == 'G':
            cov = self.distr_params.cov
        
            # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
            L = np.linalg.cholesky(cov)
            Uhat_vec = np.random.randn(mean.shape[0])
            U_vec = mean.flatten() + L @ Uhat_vec
            # U_vec = np.random.multivariate_normal(mean.flatten(), cov).T # (SLOW!! DO NOT USE)

        # sample from uniform distirbution
        elif self.distr_params.family == 'U':
            lb = self.distr_params.lb
            ub = self.distr_params.ub
            U_vec = np.random.uniform(lb.flatten(), ub.flatten()).T

        # unflatten the trajectory
        U_traj = np.zeros((2, self.ctrl_params.N-1))
        for i in range(0, self.ctrl_params.N-1):
            U_traj[:, i] = U_vec[ 2*i : 2*i+2 ]

        # saturate the inputs
        U_traj[0, :] = np.clip(U_traj[0, :], -self.sys_params.rdot_lim, self.sys_params.rdot_lim)
        U_traj[1, :] = np.clip(U_traj[1, :], -self.sys_params.thetadot_lim, self.sys_params.thetadot_lim)

        return U_traj

    # generate single integrator dynamics solution
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
        X_com_des = np.zeros((4, N))
        X_com_des[1, :] = np.ones((1, N)) * pz_des
        X_com_des[2, :] = np.ones((1, N)) * vx_des
        X_com_des[3, :] = np.zeros((1, N))

        # generate the reference trajectory
        for i in range(0, N):
            t = i * dt
            px_des = px_0 + vx_des * t
            X_com_des[0, i] = px_des[0]

        return X_com_des

    # cost function evaluation
    def cost_function(self, X_com_des, sol):
        """
        Evaulate the cost function given a propagated and desired trajecotry.
        """
        # unpack solution
        # T = sol[0]
        X_com = sol[1]
        # X_leg = sol[2]
        # P_foot = sol[3]
        # D = sol[4]
        viability = sol[5]

        # if the pendulum falls assign infinite cost
        if viability == False:
            total_cost = np.inf

        # viable solution, compute the cost
        elif viability == True:
            # unpack some cost parameters
            Q = self.ctrl_params.Q
            Qf = self.ctrl_params.Qf

            # stage cost
            state_cost = 0.0
            for i in range(0, self.N-1):
                # compute error
                xi_com = X_com[:, i]   
                xi_com_des = X_com_des[:, i]
                e_com = (xi_com - xi_com_des).reshape(4, 1)

                # compute cost
                cost = e_com.T @ Q @ e_com
                state_cost += cost

            # terminal cost
            xf_com = X_com[:, self.N-1]
            xf_com_des = X_com_des[:, self.N-1]
            ef_com = (xf_com - xf_com_des).reshape(4, 1)
            terminal_cost = ef_com.T @ Qf @ ef_com

            # total cost
            total_cost = state_cost + terminal_cost
            total_cost = total_cost[0][0]

        return total_cost
    
    # perform horizon rollouts
    def monte_carlo(self, x0, p_foot, D0):

        # preallocate the the containers
        J_list = np.zeros((1, self.K))            # cost function list
        U_list = np.zeros((2, self.N-1, self.K))  # input trajectory list
        S_list = [None for _ in range(self.K)]    # solution list

        # generate the reference trajectory
        vx_des = 0.00
        pz_des = 0.70
        X_des = self.generate_reference_trajectory(x0, pz_des, vx_des)

        # perform the rollouts
        for k in range(0, self.K):
            print('Rollout: ', k)
            # sample an input trajectory
            U = self.sample_input_trajectory()
            
            # rollout the dynamics under the input trajectory
            S = self.mdrom.RK3_rollout(x0, p_foot, U, D0)

            # evaluate the cost function, no failure, compute cost
            J = self.cost_function(X_des, S)

            # save the data
            J_list[:,k] = J
            U_list[:, :, k] = U
            S_list[k] = S

        # reorder the rollouts based on cost
        idx = np.argsort(J_list)
        idx = idx[0, :]
        J_list = J_list[:, idx]
        U_list = U_list[:, :, idx]
        S_list = [S_list[i] for i in idx]

        return S_list, U_list, J_list
    
    # perfomr sampling based predictive control
    def predictive_control(self, x0, p_foot, D0):
        """
        Perform sampling predictive control
        """
        # perform the monte carlo simulation        
        S_list, U_list, J_list = ctrl.monte_carlo(x0_com, p_foot, D0)

        # reorder the rollouts based on cost
        S_star = S_list[0]
        U_star = U_list[:, :, 0]
        J_star = J_list[0]

        # unpack the solution
        t = S_star[0]
        x_com = S_star[1]
        x_leg = S_star[2]
        p_foot = S_star[3]
        D = S_star[4]

        return t, x_com, x_leg, p_foot, D, J_star

#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":

    # decalre the system parameters
    system_params = SystemParams(m=35.0, 
                                 g=9.81, 
                                 k=5000.0, 
                                 b=250.0,
                                 l0=0.65,
                                 r_min=0.3,
                                 r_max=0.75,
                                 theta_min=-np.pi/3,
                                 theta_max=np.pi/3,
                                 rdot_lim=0.75,
                                 thetadot_lim=np.pi/3)

    # declare control parameters
    Q_diags = np.array([1.0, 1.0, 0.05, 0.05])
    Q = np.diag(Q_diags)
    Qf_diags = np.array([1.0, 1.0, 0.05, 0.05])
    Qf = np.diag(Qf_diags)
    control_params = PredictiveControlParams(N=50, 
                                             dt=0.02, 
                                             K=1500,
                                             interp='L',
                                             Q=Q,
                                             Qf=Qf)

    # declare reduced order model object
    mdrom = MDROM(system_params, control_params)

    # create parametric distribution parameters
    mean_r = 0.0          # [m/s]
    mean_theta = 0.0      # [rad/s]
    std_dev_r = 0.75      # [m/s]
    std_dev_theta = 1.0   # [rad/s]

    mean = np.array([[mean_r],              # r [m]
                     [mean_theta]])         # theta [rad]
    
    # Gaussian distribution
    std_dev = np.array([std_dev_r**2,       # r [m]
                        std_dev_theta**2])  # theta [rad]
    mean_initial = np.tile(mean, (control_params.N-1, 1))
    std_dev_matrix = np.diag(std_dev)
    I = np.eye(control_params.N-1)
    cov_initial = np.kron(I, std_dev_matrix)

    # Uniform distribution
    lb_r = -0.75    # [m/s]
    ub_r = 0.75     # [m/s]
    lb_theta = -1.0 # [rad/s]
    ub_theta = 1.0  # [rad/s]

    lb = np.array([[lb_r],      # r [m]
                   [lb_theta]]) # theta [rad]
    ub = np.array([[ub_r],      # r [m]
                   [ub_theta]]) # theta [rad]
    ones_vec = np.ones((control_params.N-1, 1))
    lb_initial = np.kron(ones_vec, lb)
    ub_initial = np.kron(ones_vec, ub)
    
    distribution_params = ParametricDistribution(family='G',
                                                 mean=mean_initial,
                                                 cov=cov_initial,
                                                 lb=lb_initial,
                                                 ub=ub_initial)
    
    # create a predictive controller object
    ctrl = PredictiveController(mdrom, system_params, control_params, distribution_params)

    # initial conditions
    x0_sys = np.array([[0.0],  # px com
                       [0.75], # pz com
                       [0.0],  # vx com
                       [0.0],  # vz com
                       [0.0],  # l0 command
                       [0.0]]) # passive angle
    p0_foot = np.array([[None], [None]])
    D0 = 'F'  # initial domain

    U = ctrl.sample_input_trajectory()
    sol = mdrom.RK3_rollout(x0_sys, p0_foot, U, D0)