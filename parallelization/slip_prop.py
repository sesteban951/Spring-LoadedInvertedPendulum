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

######################################################################################
# DYNAMICS
######################################################################################

# SLIP flight dynamics (cartesian coordinates)
def slip_flight_fwd_prop(x0, dt, alpha) -> np.array:
    """
    Simulate the flight phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # TODO: Consider adding a velocity input in the leg for flight phase.

    # system parameters
    g = 9.81   # [m/s^2] gravity
    l = 1.0    # [m]     leg free length

    # unpack the initial state
    px_0 = x0[0]
    pz_0 = x0[1]
    vx_0 = x0[2]
    vz_0 = x0[3]

    # compute the impact height and time until impact
    pz_impact = l * np.cos(alpha)
    a = 0.5 * g
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
        x_t[i, 0] = px_0 + vx_0 * t                   # pos x
        x_t[i, 1] = pz_0 + vz_0 * t - 0.5 * g * t**2  # pos z
        x_t[i, 2] = vx_0                              # vel x       
        x_t[i, 3] = vz_0 + g * t                      # vel z

# SLIP ground dynamics (polar coordinates)
def slip_ground_dyn(xk) -> np.array:

    # system parameters
    m = 10.0   # [kg]    mass
    g = 9.81   # [m/s^2] gravity
    l = 1.0    # [m]     leg free length
    k = 1000   # [N/m]   leg stiffness

    # polar state, x = [r, theta, rdot, thetadot]
    r = xk[0]
    theta = xk[1]
    r_dot = xk[2]
    theta_dot = xk[3]

    # ground phase dynamics
    xdot = np.array([
        r_dot,
        theta_dot,
        r * theta_dot**2 - g * np.cos(theta) + (k/m) * (l - r),
        -(2/r) * r_dot*theta_dot + (g/r) * np.sin(theta)
    ])

    return xdot

# SLIP ground dynamics
def slip_ground_dyn_fwd_prop(x0, dt):
    """
    Simulate the ground phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # container for the trajectory
    N = 20
    x_t = np.zeros((N, 4))

    # do RK2 integration
    xk = x0
    x_t[0, :] = xk    # TODO: check that this is flat
    for i in range(N):
        # RK2 integration
        f1 = slip_ground_dyn(x0)
        f2 = slip_ground_dyn(x0 + 0.5 * dt * f1)
        xk = xk + dt * f2
        x_t[i, :] = xk
    
    return x_t


######################################################################################
# COORDINATE TRANSFORMATIONS
######################################################################################

# Cartesian to polar coordinate
def cart_2_pol(x_cart, params, alpha) -> np.array:

    # flight state, x = [x, z, xdot, zdot]
    

######################################################################################
# MAIN
######################################################################################

if __name__ == "__main__":

    # initial state (cartesian coordinates)
    x0 = np.array([0.0,   # px
                   0.0,   # pz
                   0.3,   # vx
                   0.01]) # vz
    dt = 0.01
    alpha = 0.0

    # simulate the SLIP model
    slip_flight_dyn(x0, dt, alpha)

    # initial state (polar coordinates)
    x0 = np.array([1.1,   # r
                   0.0,   # theta
                   0.0,   # rdot
                   0.5]) # thetadot
    slip_ground_dyn(x0, dt)    
