
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
# @partial(jit, static_argnums=(2, 4))
def slip_flight_fwd_prop(x0: jnp.array, 
                         alpha: float,
                         dt: float,
                         apex_terminate: bool,
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

    # ensure that the COM is above the ground
    assert pz_0 > 0, f"The center of mass is under ground. pz = {pz_0}"

    # compute time until apex
    if vz_0 >= 0:
        t_apex = vz_0 / params.g  # from vz(t) = vz_0 - g*t
    else:
        t_apex = None

    # Apex Condition: if you want to terminate at the apex
    if apex_terminate is True:
        # find the zero z-velocity time
        assert vz_0 >= 0, "The SLIP z-velocity is negative (past the apex), therefore no such positive apex time exists."
        t_terminate = t_apex

    # Guard Condition: compute the time until impact
    else:
        pz_impact = params.l0 * jnp.cos(alpha)
        a = 0.5 * params.g
        b = -vz_0
        c = -(pz_0 - pz_impact)
        s = jnp.sqrt(b**2 - 4 * a * c)
        d = 2 * a
        t1 = (-b + s) / d
        t2 = (-b - s) / d
        t_terminate = jnp.maximum(t1, t2)
        assert t_terminate >= 0, "No impact time exists. Time until impact must be positive or equal to zero."
        # TODO: implement error handling with bad alphas

    # create a time vector
    t_span = jnp.arange(0, t_terminate, dt)
    t_span = jnp.append(t_span, t_terminate)
    t_span = t_span.reshape(-1, 1)

    # create a trajectory vector
    x_t = jnp.zeros((len(t_span), 4))

    # there exists apex state
    if t_apex is not None:   
        x_apex = jnp.array([px_0 + vx_0 * t_apex,
                            pz_0 + vz_0 * t_apex - 0.5 * params.g * t_apex**2,
                            vx_0,
                            vz_0 - params.g * t_apex])
    # there does not exist apex state
    else:                   
        print("No apex state exists. Returning zeros vector.")
        x_apex = jnp.array([0, 0, 0, 0])

    # simulate the flight phase
    for i in range(len(t_span)):
        # update the state
        t = t_span[i, 0]
        x_t = x_t.at[i, 0].set(px_0 + vx_0 * t)                          # pos x
        x_t = x_t.at[i, 1].set(pz_0 + vz_0 * t - 0.5 * params.g * t**2)  # pos z
        x_t = x_t.at[i, 2].set(vx_0)                                     # vel x       
        x_t = x_t.at[i, 3].set(vz_0 - params.g * t)                      # vel z

        # check that the COM is above the ground
        assert pz_0 > 0, f"The center of mass is under ground. pz = {pz_0}"

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
    t_span, x_t, u_t, alpha, x_apex, D_t, p_foot = slip_flight_fwd_prop(x0, alpha, dt, apex_terminate, sys_params)

    # TODO: not returning full arrays somtimes
    t_span.block_until_ready()
    x_t.block_until_ready()
    u_t.block_until_ready()
    x_apex.block_until_ready()
    D_t.block_until_ready()
    p_foot.block_until_ready()

    t_tot = 0.0
    for i in range(10):
        
        t0 = time.time()
        t_span, x_t, u_t, alpha, x_apex, D_t, p_foot = slip_flight_fwd_prop(x0, alpha, dt, apex_terminate, sys_params)
        
        # TODO: not returning full arrays somtimes
        t_span.block_until_ready()
        x_t.block_until_ready()
        u_t.block_until_ready()
        x_apex.block_until_ready()
        D_t.block_until_ready()
        p_foot.block_until_ready()
        
        t1 = time.time()
        dt = t1 - t0
        print(f"Elapsed time: {dt}")
        t_tot += dt

    print(f"Average time: {t_tot/10}")

    # plot the results
    plt.figure()
    plt.plot(x_t[:,0], x_t[:,1], 'b.')
    plt.plot(p_foot[0], p_foot[1], 'ro')
    plt.grid()
    plt.xlabel('px [m]')
    plt.ylabel('pz [m]')
    plt.show()
