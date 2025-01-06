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
        self.l0 = system_params.l0
        self.k = system_params.k
        self.b = system_params.b

        # initialize control parameters
        self.dt = control_params.dt
        self.N = control_params.N
        self.interp = control_params.interp

    #################################  DYNAMICS  #################################

    def dynamics(self, x_com, p_foot, u, d):
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

        # Ground stance domain (G)
        elif d == 'G':
            # compute the leg state
            r_vec = p_foot - p_com
            r_norm = np.linalg.norm(r_vec)
            r_hat = r_vec / r_norm

            # get the control input
            v = u[0]
            u = k * (v - l0)

            # compute the dynamics
            a_leg = -(r_hat/m) * (k * (l0 - r_norm) + b * (v_com.T @ r_hat) + u)
            a_com = a_leg + np.array([[0], [-g]])
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
    def RK3_rollout(self, x0_com, p_foot, U, D0):
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
        xt_leg = np.zeros((4, N))
        pt_foot = np.zeros((2, N))
        D = [None for _ in range(N)]

        # initial conditions
        x_leg = self.update_leg_state(x0_com, p_foot, U[:, 0], D0)
        xt_leg[:, 0] = x_leg.reshape(4)
        xt_com[:, 0] = x0_com.reshape(4) 
        pt_foot[:, 0] = p_foot.reshape(2)
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
                               p_foot,
                               u1, 
                               Dk)
            f2 = self.dynamics(xk_com + dt/2 * f1, 
                               p_foot, 
                               u2, 
                               Dk)
            f3 = self.dynamics(xk_com - dt*f1 + 2*dt*f2, 
                               p_foot,
                               u3, 
                               Dk)

            # RK3 step and update all states
            xk_com = xk_com + (dt/6) * (f1 + 4*f2 + f3)
            xk_leg = self.update_leg_state(xk_com, p_foot, u1, Dk)

            # check if hit switching surfaces and update state if needed
            checked_contacts = self.check_switching(xk_com, xk_leg, u1, contacts)
            
            # if a change in contact detected
            if checked_contacts != contacts:
                
                # update the feet positions
                p_foot = self.apply_reset(xk_com, xk_leg, checked_contacts)
                
                # update domain and contact info
                Dk = self.domain_identification(checked_contacts)
                contacts = checked_contacts

            # store the states
            xt_com[:, i+1] = xk_com.reshape(4)
            xt_leg[:, i+1] = xk_leg.reshape(4)
            
            # store the foot positions
            if Dk == 'F':
                pt_foot[:, i+1] = self.get_foot_pos(xk_com, xk_leg, pz_zero=False).flatten()
            elif Dk == 'G':
                pt_foot[:, i+1] = p_foot.reshape(2)

            # store the domain
            D[i+1] = Dk

        return Tx, xt_com, xt_leg, pt_foot,  D
    
    #################################  STATE UPDATES  #################################

    # update leg polar state
    def update_leg_state(self, x_com, p_foot, u, D):
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
            r = u[0]
            theta = u[1]

            # pack into state vectors
            x_leg = np.array([[r],
                              [theta],
                              [0.0],
                              [0.0]])
        
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
    def check_switching(self, x_com, x_leg, u, contact):
        """
        Check if the state has hit any switching surfaces
        """

         # In Flight (F)
        if contact == False:
            # check for touchdown
            contact_result = self.S_TD(x_com, x_leg)
        # In Ground(G)
        elif contact == True:
            # check for takeoff
            u_leg = u[0]
            contact_result = self.S_TO(x_com, x_leg, u_leg)

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
    def S_TO(self, x_com, x_leg, u_leg):
        """
        Check if a leg has taken off from the ground
        """
        # get relevant states
        r = x_leg[0]
        rdot = x_leg[2]

        # check the switching surface conditions
        # set by input
        nom_length = (r >= u_leg)      # the leg is at its nominal uncompressed length
        # nom_length = (r >= self.l0)      # the leg is at its nominal uncompressed length
        pos_vel = (rdot >= 0.0)          # the leg is going in uncompressing direction
        takeoff = nom_length and pos_vel # if true, leg has taken off into flight

        # return the contact result
        if takeoff==True:
            contact = False
        elif takeoff==False:
            contact = True

        return contact

    # apply a reset map
    def apply_reset(self, x_com, x_leg, contact):

        # Left leg update
        if contact == True:
            p_foot = self.get_foot_pos(x_com, x_leg, pz_zero=True)
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
        Simple contact identification
        """
        # Flight (F)
        if D == 'F':
            contact = False
        # Ground Support (G)
        elif D == 'G':
            contact = True
        
        return contact

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
    control_params = PredictiveControlParams(N=175, 
                                             dt=0.01, 
                                             K=100,
                                             interp='Z')

    # declare reduced order model object
    mdrom = MDROM(system_params, control_params)

    # initial conditions
    x0_com = np.array([[0.25], # px [m]
                       [1.5], # py [m]
                       [0.05],  # vx [m/s]
                       [1]]) # vz [m/s]
    p_foot = np.array([[None],  # px [m]
                       [None]]) # py [m]
    D0 = 'F'

    # CONSTANT INPUT
    u_constant = np.array([[system_params.l0 * 1.0], # left leg
                           [0]]) # right leg
    U = np.tile(u_constant, (1, control_params.N-1))

    # run the simulation
    t0 = time.time()
    t, x_com, x_leg, p_foot, D = mdrom.RK3_rollout(x0_com, p_foot, U, D0)
    tf = time.time()
    print('Simulation time: ', tf - t0)

    # print(t.shape)
    # print(x_com.shape)
    # print(x_left.shape)
    # print(x_right.shape)
    # print(p_left.shape)
    # print(p_right.shape)
    # print(len(D))

    print(p_foot.T)

    # save the data into CSV files
    np.savetxt('./data/single/time.csv', t, delimiter=',')
    np.savetxt('./data/single/state_com.csv', x_com.T, delimiter=',')
    np.savetxt('./data/single/state_leg.csv', x_leg.T, delimiter=',')
    np.savetxt('./data/single/pos_foot.csv', p_foot.T, delimiter=',')
    np.savetxt('./data/single/domain.csv', D, delimiter=',', fmt='%s')
