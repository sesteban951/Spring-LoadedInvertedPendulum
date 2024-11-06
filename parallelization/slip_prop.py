import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from dataclasses import dataclass

######################################################################################
# STRUCTS
######################################################################################

# data class to hold the system parameters
@dataclass
class slip_params:
    m:  float  # mass, [kg]
    l0: float  # leg free length, [m]
    k:  float  # leg stiffness, [N/m]
    g:  float  # gravity, [m/s^2]

######################################################################################
# DYNAMICS
######################################################################################

# SLIP ground dynamics (polar coordinates)
def slip_ground_dyn(xk: np.array, 
                    params: slip_params) -> np.array:
    """
    Closed form dynamics for the ground phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # polar state, x = [r, theta, rdot, thetadot]
    r = xk[0]
    theta = xk[1]
    r_dot = xk[2]
    theta_dot = xk[3]

    # ground phase dynamics
    xdot = np.array([
        r_dot,
        theta_dot,
        r * theta_dot**2 - params.g * np.cos(theta) + (params.k/params.m) * (params.l0 - r),
        -(2/r) * r_dot*theta_dot + (params.g/r) * np.sin(theta)
    ])

    return xdot

# SLIP ground dynamics (polar coordinates)
def slip_ground_fwd_prop(x0: np.array, 
                         dt: float, 
                         params:float) -> np.array:
    """
    Simulate the ground phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # ensure the SLIP hasn't fallen over
    assert (abs(x0[1]) <= np.pi/2), "The SLIP has fallen over in ground phase."

    # container for the trajectory, TODO: figure out a static size for the container. Dynamic = bad
    x_t = []
    x_t.append(x0)    

    # do Integration until switching conditions are met
    k = 0
    xk = x0
    dot_product, leg_uncompressed = False, False
    while not (leg_uncompressed * dot_product): # TODO: consider better switching detection

        # RK2 integration (in polar coordinates)
        f1 = slip_ground_dyn(xk, params)
        f2 = slip_ground_dyn(xk + 0.5 * dt * f1, params)
        xk = xk + dt * f2
        x_t.append(xk) #  TODO: figure out a static size for the container. Dynamic = bad

        # check take-off guard conditions
        x_cart = polar_to_cartesian(xk, params)  # consider just saving this vector directly to save on compute
        leg_pos = np.array([x_cart[0], x_cart[1]])
        leg_vel = np.array([x_cart[2], x_cart[3]])
        dot_product = (np.dot(leg_pos, leg_vel) >= 0)
        leg_uncompressed = (xk[0] >= params.l0)

        # increment the counter
        k += 1

    # convert the trajectory to a numpy array
    N = len(x_t)
    x_t = np.array(x_t)

    # domain information (1 for ground phase)
    D_t = np.ones((N, 1))

    # time span infomration
    t_span = np.linspace(0, (N-1)*dt, N)

    return t_span, x_t, D_t

# SLIP flight dynamics (cartesian coordinates)
def slip_flight_fwd_prop(x0: np.array, 
                         dt: float, 
                         apex_terminate: bool,
                         params: slip_params) -> np.array:
    """
    Simulate the flight phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # TODO: Consider adding a velocity input in the leg for flight phase.

    # unpack the initial state
    px_0 = x0[0]  # these are in world frame
    pz_0 = x0[1]  # these are in world frame
    vx_0 = x0[2]
    vz_0 = x0[3]

    # ensure that the COM is above the ground
    assert pz_0 > 0, "The center of mass is under ground. pz = ".format(pz_0)

    # choose control action
    vx_des = 0.0
    alpha = raibert_controller(x0, vx_des, params)

    # Apex Condition: compute time until apex
    if apex_terminate is True:
        # find the zero velocity time
        assert vz_0 > 0, "The SLIP z-velocity is not upwwards, therfore no such positive apex time exists."
        t_terminate = vz_0 /params.g  # from vz(t) = vz_0 - g*t

    # Guard Condition: compute the time until impact
    else:
        pz_impact = params.l0 * np.cos(alpha)
        a = 0.5 * params.g
        b = - vz_0
        c = -(pz_0 - pz_impact)
        s = np.sqrt(b**2 - 4*a*c)
        d = 2 * a
        t1 = (-b + s) / d
        t2 = (-b - s) / d
        t_terminate = max(t1, t2)
        assert t_terminate >= 0, "Time until impact must be positive or equal to zero."

    # create a time vector
    t_span = np.arange(0, t_terminate, dt)
    t_span = np.append(t_span, t_terminate)

    # create a trajectory vector
    x_t = np.zeros((len(t_span), 4))

    # simulate the flight phase
    for i, t in enumerate(t_span):

        # update the state
        x_t[i, 0] = px_0 + vx_0 * t                          # pos x
        x_t[i, 1] = pz_0 + vz_0 * t - 0.5 * params.g * t**2  # pos z
        x_t[i, 2] = vx_0                                     # vel x       
        x_t[i, 3] = vz_0 - params.g * t                      # vel z

        # check that the COM is above the ground
        assert pz_0 > 0, "The center of mass is under ground. pz = ".format(pz_0)

    # Domain information (0 for flight phase)
    D_t = np.zeros((len(t_span), 1))

    # compute the final leg position in world frame
    x_com = x_t[-1, :]     # take the last state as the final state
    p_foot = np.array([x_com[0] - params.l0 * np.sin(alpha),
                       x_com[1] - params.l0 * np.cos(alpha)])

    return t_span, x_t, D_t, p_foot

######################################################################################
# CONTROL
######################################################################################

# simple raibert controller # TODO: this will be a normal distirbution
def raibert_controller(x_flight: np.array,
                       v_des: float,
                       params: slip_params) -> float:
    """
    Simple Raibert controller for the SLIP model.
    """
    # unpack the state
    vx = x_flight[2]

    # compute the desired angle
    kd = 0.15
    alpha = -kd * (vx - v_des) 

    return alpha

######################################################################################
# COORDINATE TRANSFORMATIONS
######################################################################################

# Polar to Cartesian coordiante
def polar_to_cartesian(x_polar: np.array, 
                       params: slip_params) -> np.array:
    """
    Convert the polar coordinates to cartesian coordinates.
    Assumes the foot is at (0,0), must add the last leg pos after this method to keep track in world frame
    """
    # polar state, x = [r, theta, rdot, thetadot]
    r = x_polar[0]
    theta = x_polar[1]
    r_dot = x_polar[2]
    theta_dot = x_polar[3]

    # full state in cartesian coordintes
    px = r * np.sin(theta)      # COM position x
    pz = r * np.cos(theta)      # COM position z
    vx = r_dot * np.sin(theta) + r * theta_dot * np.cos(theta)  # COM velocity x
    vz = r_dot * np.cos(theta) - r * theta_dot * np.sin(theta)  # COM velocity z

    return np.array([px, pz, vx, vz])

# Cartesian to polar coordinate
def carteisan_to_polar(x_cart_local: np.array, 
                       params: slip_params) -> np.array:
    """
    Convert the cartesian coordinates to polar coordinates.
    Assumes the cartesian coordinates are local:    x_cart = x_com_W = x_foot_W 
    """
    # flight state, x = [x, z, xdot, zdot]
    px = x_cart_local[0]
    pz = x_cart_local[1]
    vx = x_cart_local[2]
    vz = x_cart_local[3]

    # full state in polar coordinates
    r = np.sqrt(px**2 + pz**2)           # leg length
    th = np.arctan2(px, pz)              # leg angle
    r_dot = (px * vx + pz * vz) / r      # leg length rate
    th_dot = (vx * pz - px * vz) / r**2  # leg angle rate

    return np.array([r, th, r_dot, th_dot])

# get the foot position given com pos and attack angle
def cartesian_local_flight(x_cart_W: np.array, 
                           alpha: float,
                           params: slip_params) -> np.array:
    """
    Compute the COM w.r.t. foot pos. When the leg is in the air and uncompressed.
    """
    # compute the COM position
    x0_cart_local = np.array([params.l0 * np.sin(alpha), 
                              params.l0 * np.cos(alpha),
                              x_cart_W[2],
                              x_cart_W[3]])
    return x0_cart_local

######################################################################################
# MAIN
######################################################################################

if __name__ == "__main__":

    # define the sytem parameters
    sys_params = slip_params(m  = 1.0,   # mass [kg]
                             l0 = 1.0,   # leg free length [m]
                             k  = 100.0,  # leg stiffness [N/m]
                             g  = 9.81)  # gravity [m/s^2]

    # simulation parameters
    dt = 0.005
    apex_terminate = False

    # initial state in cartesian
    x0_cart_global = np.array([0.0, 
                               3.0, 
                               1.0, 
                               1.0])
    t_span, x_t, D_t, p_foot = slip_flight_fwd_prop(x0_cart_global, dt, apex_terminate, sys_params)

    # plot the results 
    plt.plot(x_t[:,0], x_t[:,1], 'b.')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.show()

