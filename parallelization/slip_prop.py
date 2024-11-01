import numpy as np
import scipy as sp

import jax
from jax import lax
from jax.lib import xla_bridge
from jax import jit, vmap
import jax.numpy as jnp

from functools import partial
import time
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
def slip_ground_dyn(xk, params) -> np.array:
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

# SLIP ground dynamics
def slip_ground_dyn_fwd_prop(x0, dt, params) -> np.array:
    """
    Simulate the ground phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # container for the trajectory
    N = 200
    t_span = np.arange(0, N*dt, dt)
    x_t = np.zeros((N, 4))

    # do RK2 integration
    xk = x0
    x_t[0, :] = xk    # TODO: check that this is flat
    for i in range(N):
        # RK2 integration
        f1 = slip_ground_dyn(x0, params)
        f2 = slip_ground_dyn(x0 + 0.5 * dt * f1, params)
        xk = xk + dt * f2
        x_t[i, :] = xk

    # domain information (1 for ground phase)
    D_t = np.ones((N, 1))

    return t_span, x_t, D_t

# SLIP flight dynamics (cartesian coordinates)
def slip_flight_fwd_prop(x0, dt, alpha, params) -> np.array:
    """
    Simulate the flight phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # TODO: Consider adding a velocity input in the leg for flight phase.

    # unpack the initial state
    px_0 = x0[0]
    pz_0 = x0[1]
    vx_0 = x0[2]
    vz_0 = x0[3]

    # compute the impact height and time until impact
    pz_impact = params.l0 * np.cos(alpha)
    a = 0.5 * params.g
    b = - vz_0
    c = -(pz_0 - pz_impact)
    s = np.sqrt(b**2 - 4*a*c)
    d = 2 * a
    t1 = (-b + s) / d
    t2 = (-b - s) / d
    t_impact = max(t1, t2)
    assert t_impact > 0, "Time until impact must be positive."

    # create a time vector
    t_span = np.arange(0, t_impact, dt)
    t_span = np.append(t_span, t_impact)

    # create a trajectory vector
    x_t = np.zeros((len(t_span), 4))
    for i, t in enumerate(t_span):
        x_t[i, 0] = px_0 + vx_0 * t                              # pos x
        x_t[i, 1] = pz_0 + vz_0 * t - 0.5 * params.g * t**2  # pos z
        x_t[i, 2] = vx_0                                         # vel x       
        x_t[i, 3] = vz_0 + params.g * t                      # vel z

    # Domain information (0 for flight phase)
    D_t = np.zeros((len(t_span), 1))

    return t_span, x_t, D_t

######################################################################################
# COORDINATE TRANSFORMATIONS
######################################################################################

# Cartesian to polar coordinate
def carteisan_to_polar(x_cart, alpha, params) -> np.array:

    # flight state, x = [x, z, xdot, zdot]
    px = x_cart[0]
    pz = x_cart[1]
    vx = x_cart[2]
    vz = x_cart[3]

    # positions
    px_com = px    # x com position in world frame
    pz_com = pz    # z com position in world frame
    px_foot = px + params.l0 * np.sin(alpha)  # foot position x
    pz_foot = pz - params.l0 * np.cos(alpha)  # foot position z

    x = px_com - px_foot
    z = pz_com - pz_foot

    # full state in polar coordinates
    r = np.sqrt(x**2 + z**2)       # leg length
    th = np.arctan2(x, z)          # leg angle
    r_dot = (x * vx + z * vz) / r  # leg length rate
    th_dot = (z * vx - x * vz) / r**2  # leg angle rate

    return np.array([r, th, r_dot, th_dot])

def polar_to_cartesian(x_polar, params) -> np.array:

    # polar state, x = [r, theta, rdot, thetadot]
    r = x_polar[0]
    theta = x_polar[1]
    r_dot = x_polar[2]
    theta_dot = x_polar[3]

    # full state in cartesian coordintes
    px = r * np.sin(theta)  # COM position x
    pz = r * np.cos(theta)  # COM position z
    vx = r_dot * np.sin(theta) + r * theta_dot * np.cos(theta)  # COM velocity x
    vz = r_dot * np.cos(theta) - r * theta_dot * np.sin(theta)  # COM velocity z

    return np.array([px, pz, vx, vz])

######################################################################################
# MAIN
######################################################################################

if __name__ == "__main__":

    # define the sytem parameters
    sys_params = slip_params(m  = 10.0,   # mass [kg]
                             l0 = 1.0,    # leg free length [m]
                             k  = 1000.0, # leg stiffness [N/m]
                             g  = 9.81)   # gravity [m/s^2]

    # initial state (cartesian coordinates)
    # x0 = np.array([0.0,   # px
    #                3.0,   # pz
    #                0.3,   # vx
    #                0.01]) # vz
    # dt = 0.01
    # alpha = 0.0
    # t_flight, xt_flight, D_t = slip_flight_fwd_prop(x0, dt, alpha, sys_params)

    # initial state (polar coordinates)
    x0 = np.array([1.0,   # r
                   0.0,   # theta
                   0.0,   # rdot
                   0.5]) # thetadot
    dt = 0.01
    t_span, x_t, D_t = slip_ground_dyn_fwd_prop(x0, dt, sys_params)

    # convert the polar coordinates to cartesian
    for i in range(len(t_span)):
        x_t[i, :] = polar_to_cartesian(x_t[i, :], sys_params)

    # plot the positoins
    plt.figure()
    plt.plot(x_t[:, 0], x_t[:, 1])
    plt.plot(x_t[0, 0], x_t[0, 1], 'go')
    plt.plot(x_t[-1, 0], x_t[-1, 1], 'rx')
    plt.plot(0, 0, 'ko')
    plt.show()