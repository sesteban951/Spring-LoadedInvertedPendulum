
import numpy as np
import matplotlib.pyplot as plt
import time

import jax
from jax import lax
from jax.lib import xla_bridge
from jax import jit, vmap
import jax.numpy as jnp

from functools import partial  # enforcing partial static arguments
from flax import struct        # enforcing immutable data structures

######################################################################################
# STRUCTS
######################################################################################

# data class to hold the system parameters
@struct.dataclass
class slip_params:
    m:  float    # mass, [kg]
    l0: float    # leg free length, [m]
    k:  float    # leg stiffness, [N/m]
    br: float    # radial damping coeff, [Ns/m]
    ba: float    # angular damping coeff, [Ns/m]
    g:  float    # gravity, [m/s^2]
    amax: float  # max angle control input, [rad]
    amin: float  # min angle control input, [rad]
    umax: float  # max control input, [N]
    umin: float  # min control input, [N]
    dt:  float   # integration time step, [s]

######################################################################################
# DYNAMICS
######################################################################################

# SLIP flight forward propagation
@partial(jit, static_argnums=(2))
def slip_flight_fwd_prop(x0: jnp.array, 
                         alpha: float, 
                         params: slip_params):
    '''
    Simulate the flight phase of the SLIP model
    '''
    # unpack the initial state
    px = x0[0]  # x-pos in world frame
    pz = x0[1]  # z-pos in world frame
    vx = x0[2]
    vz = x0[3]

    # TODO: the branch here is not compatible with JAX
    # check that the intial z-velocity is positive
    msg = "Intial Condition Error: z-vel must be greater than zero."
    def raise_error(x):
        return x * (-jnp.inf)
    def pass_through(x):
        return x
    res = lax.cond(vz < 0, raise_error, pass_through, vz) 

    # compute the time until apex and impact, (vz(t) = vz_0 - g*t)
    t_apex = vz / params.g

    # compute the impact time,  (pz(t) = pz_0 + vz_0*t - 0.5*g*t^2)
    #                        => 0 = (pz_0 - pz_impact) + vz_0*t_impact - 0.5*g*t_impact^2
    pz_impact = params.l0 * jnp.cos(alpha)
    a = -0.5 * params.g
    b = vz
    c = pz - pz_impact
    s = jnp.sqrt(b**2 - 4*a*c)
    t1 = (-b + s) / (2*a)
    t2 = (-b - s) / (2*a)
    t_impact = jnp.maximum(t1, t2)

    # compute the apex state
    x_apex = jnp.zeros(4)
    x_apex = x_apex.at[0].set(px + vx * t_apex)
    x_apex = x_apex.at[1].set(pz + vz * t_apex - 0.5 * params.g * t_apex**2)
    x_apex = x_apex.at[2].set(vx)
    x_apex = x_apex.at[3].set(vz - params.g * t_apex)

    # Compute the impact state
    x_impact = jnp.zeros(4)
    x_impact = x_impact.at[0].set(px + vx * t_impact)
    x_impact = x_impact.at[1].set(pz + vz * t_impact - 0.5 * params.g * t_impact**2)
    x_impact = x_impact.at[2].set(vx)
    x_impact = x_impact.at[3].set(vz - params.g * t_impact)

    return t_apex, t_impact, x_apex, x_impact

# SLIP ground dynamics
@ partial(jit, static_argnums=(2))
def slip_ground_dynamics(x: jnp.array,
                         u: float,
                         params: slip_params):
    '''
    Ground phase dynamics vector field
    '''
    # unpack the state
    r = x[0]         # leg length
    theta = x[1]     # leg angle
    r_dot = x[2]     # leg length rate
    theta_dot = x[3] # leg angle rate

    # ground phase dynamics
    xdot = jnp.zeros(4)
    xdot = xdot.at[0].set(r_dot)
    xdot = xdot.at[1].set(theta_dot)
    xdot = xdot.at[2].set(r * theta_dot**2 - params.g*jnp.cos(theta) + (params.k/params.m)*(params.l0 - r) 
                          - (params.br/params.m)*r_dot + (1/params.m) * u)
    xdot = xdot.at[3].set(-(2/r) * r_dot*theta_dot + (params.g/r) * jnp.sin(theta) 
                          - (params.ba/params.m)*theta_dot)

    return xdot

# SLIP ground forward propagation
# @partial(jit, static_argnums=(2))
def slip_ground_fwd_prop(x0: jnp.array,
                         u: jnp.array,
                         params: slip_params):
        '''
        Simulate the ground phase of the SLIP model
        '''
        # unpack the initial state
        r = x0[0]         # leg length
        theta = x0[1]     # leg angle
        r_dot = x0[2]     # leg length rate
        theta_dot = x0[3] # leg angle rate


######################################################################################
# TESTING
######################################################################################

# just printing some info here
def print_platform_info() -> None:

    print(50*"*")

    # Print JAX version
    print(f"JAX version: {jax.__version__}")

    # Print platform (e.g., 'cpu', 'gpu', or 'tpu')
    hardware = xla_bridge.get_backend().platform
    print(f"Hardware: {hardware}")

    # Print device information
    devices = jax.devices()
    for device in devices:
        print(f"Device: {device.device_kind} - {device}")

    print(50*"*")

######################################################################################
# MAIN
######################################################################################

if __name__ == "__main__":

    # check that the GPU is available
    print_platform_info()

    ######################################################################################

    # Define the system parameters with jnp.float32 for compatibility with JAX
    sys_params = slip_params(m  = float(1.0),     # mass [kg]
                             l0 = float(1.0),     # leg free length [m]
                             k  = float(500.0),   # leg stiffness [N/m]
                             br = float(0.0),     # radial damping [Ns/m]
                             ba = float(0.0),     # angular damping [Ns/m]
                             g  = float(9.81),    # gravity [m/s^2]
                             amax = float(1.5),   # max angle control input [rad]
                             amin = float(-1.5),  # min angle control input [rad]
                             umax = float(10.0),  # max control input [N]
                             umin = float(-10.0), # min control input [N]
                             dt = float(0.005)    # time step [s]
                             )

    # intial condition
    x0 = jnp.array([0,     # px
                    3.0,   # pz
                    1.0,   # vx
                    0.5])  # vz

    t_apex, t_impact, x_apex, x_impact = slip_flight_fwd_prop(x0, 0.0, sys_params)
    print(f"t_apex: {t_apex}")
    print(f"t_impact: {t_impact}")
    print(f"x_apex: {x_apex}")
    print(f"x_impact: {x_impact}")
