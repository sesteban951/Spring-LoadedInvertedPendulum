
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
    umax: float  # max control input, [N]
    umin: float  # min control input, [N]

######################################################################################
# DYNAMICS
######################################################################################

# SLIP flight dynamics (returns in cartesian world frame)
@partial(jit, static_argnums=(2, 3))
def slip_flight_fwd_prop(x0: jnp.array, 
                         alpha: float,
                         dt: float,
                         params: slip_params) -> jnp.array:
    """
    Simulate the flight phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # TODO: Consider adding a velocity input in the leg for flight phase.

    # unpack the initial state
    px_0 = x0[0]  # these are in world frame
    pz_0 = x0[1]  # these are in world frame
    vx_0 = x0[2]
    vz_0 = x0[3]

    # Apex condition: compute the apex state, from vz(t) = vz_0 - g*t = 0
    # NOTE: assumes we are not passed the apex
    t_apex = vz_0 / params.g 
    x_apex = jnp.array([px_0 + vx_0 * t_apex,
                        pz_0 + vz_0 * t_apex - 0.5 * params.g * t_apex**2,
                        vx_0,
                        0.0])
    
    # compute the time until impact
    pz_impact = params.l0 * jnp.cos(alpha)
    a = 0.5 * params.g
    b = vz_0
    c = -(pz_0 - pz_impact)
    s = jnp.sqrt(b**2 - 4*a*c)
    t1 = (-b + s) / (2*a)
    t2 = (-b - s) / (2*a)
    t_terminate = jnp.where(t1 > 0, t1, t2)
    # t_terminate = lax.stop_gradient(jnp.where(t1 > 0, t1, t2))

    # generate the time span
    t_span = jnp.arange(0.0, t_terminate, dt)
    # t_span = jnp.append(t_span, t_terminate)
    exit(0)

    # generate a control input vector (should be zeros since no control in flight phase)
    u_t = jnp.zeros((len(t_span), 1))

    # Domain information (0 for flight phase)
    D_t = jnp.zeros((len(t_span), 1))

    # compute the final leg position in world frame
    x_com = x_t[-1, :]     # take the last state as the final state
    p_foot = jnp.array([x_com[0] - params.l0 * jnp.sin(alpha),
                        x_com[1] - params.l0 * jnp.cos(alpha)])

    return t_span, x_t, u_t, alpha, x_apex, D_t, p_foot

######################################################################################
# CONTROL
######################################################################################

# simple leg landing controller # TODO: this will be a normal distribution
def angle_control(x_flight: jnp.array,
                  v_des: float,
                  params: slip_params) -> float:
    """
    Simple Raibert controller for the SLIP model.
    """
    # unpack the state
    vx = x_flight[2]

    # compute the desired angle from simple Raibert controller
    kd = 0.13
    alpha = -kd * (vx - v_des) 

    return alpha

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
                             umax = float(10.0),  # max control input [N]
                             umin = float(-10.0)  # min control input [N]
                            )

    # intial condition
    x0 = jnp.array([0,  # px
                    3.0,  # pz
                    1.0,  # vx
                    0.5]) # vz
    dt = 0.005
    apex_terminate = False

    # simulate the flight phase
    alpha = 0.0
    t_span, x_t, u_t, alpha, x_apex, D_t, p_foot = slip_flight_fwd_prop(x0, alpha, dt, sys_params)

