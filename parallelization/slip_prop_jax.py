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

# system parameters data class
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

# normal distribution data class
@struct.dataclass
class normal_dist:
    mean: jnp.array
    cov: jnp.array

# uniform distribution data class
@ struct.dataclass
class uniform_dist:
    lower: jnp.array
    upper: jnp.array

######################################################################################
# DYNAMICS
######################################################################################

# SLIP flight forward propagation
@partial(jit, static_argnums=(2))
def slip_flight_fwd_prop(x0: jnp.array, 
                         distribution: uniform_dist,
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

    # compute the angle of the leg at apex
    alpha = 0.0 # TODO: sample this from a distribution

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
    x_apex = jnp.array([px + vx * t_apex,
                        pz + vz * t_apex - 0.5 * params.g * t_apex**2,
                        vx,
                        vz - params.g * t_apex])

    # Compute the impact state
    x_impact = jnp.array([px + vx * t_impact,
                          pz + vz * t_impact - 0.5 * params.g * t_impact**2,
                          vx,
                          vz - params.g * t_impact])

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
    xdot = jnp.array([r_dot,
                      theta_dot,
                      r * theta_dot**2 - params.g * jnp.cos(theta) + (params.k / params.m) * (params.l0 - r) 
                      - (params.br / params.m) * r_dot + (1 / params.m) * u,
                      -(2 / r) * r_dot * theta_dot + (params.g / r) * jnp.sin(theta) 
                      - (params.ba / params.m) * theta_dot])
    
    return xdot

# SLIP ground forward propagation
@partial(jit, static_argnums=(1))
def slip_ground_fwd_prop(x0: jnp.array,
                         params: slip_params):
    '''
    Simulate the ground phase of the SLIP model
    '''

    # boolean function to stop while loop
    def _condition_fun(args):

        # unpack the arguments
        _, _, _, _, take_off = args

        return ~take_off

    # RK2 integration
    def _RK2_step(args):

        # unpack the arguments
        i, x_prev, u_t, history, take_off = args

        # get the current control input
        u = u_t[i]

        # do the RK2 step        
        f1 = slip_ground_dynamics(x_prev, u, params)
        f2 = slip_ground_dynamics(x_prev + 0.5 * params.dt * f1, u, params)
        x_next = x_prev + params.dt * f2

        # print the counter
        i += 1

        # append to the history
        history = history.at[i].set(x_next)

        # check if the condition to stop is met
        r = x_next[0]
        theta = x_next[1]
        r_dot = x_next[2]
        theta_dot = x_next[3]
        px = r * jnp.sin(theta) 
        pz = r * jnp.cos(theta)
        vx = r_dot * jnp.sin(theta) + r * theta_dot * jnp.cos(theta)
        vz = r_dot * jnp.cos(theta) - r * theta_dot * jnp.sin(theta)
        
        # vectors to use for the next iteration
        l_leg = jnp.array([px, pz])
        v_com = jnp.array([vx, vz])

        # booleans
        leg_uncompressed = (r >= params.l0)
        ortho_velocity = (jnp.dot(v_com, l_leg) >= 0)

        # check if have hit the switching surface
        take_off = leg_uncompressed & ortho_velocity

        return (i, x_next, u_t, history, take_off)

    # define max iterations for the while loop
    max_iters = 250

    # define a input signal, counting from 0 to max_iters
    u_t = jnp.zeros(max_iters)
    x_t = jnp.full((max_iters,4), jnp.nan)
    x_t = x_t.at[0].set(x0)
    res = lax.while_loop(_condition_fun,
                         _RK2_step,
                         (0, x0, u_t, x_t, False))

    return res

######################################################################################
# COORDINATE CONVERSION
######################################################################################

# Polar to Cartesian
@partial(jit, static_argnums=(2))
def polar_to_cartesian(x_polar: np.array,
                       p_foot_W: np.array,
                       sys_params: slip_params) -> np.array:
    """
    Convert the polar coordinates to cartesian coordinates.
    """
    # polar state, x = [r, theta, r_dot, theta_dot]
    r = x_polar[0]
    theta = x_polar[1]
    r_dot = x_polar[2]
    theta_dot = x_polar[3]

    # foot x-position
    px_foot = p_foot_W[0]

    # full state in cartesian coordiantes
    x_cartesian = jnp.array([r * jnp.sin(theta) + px_foot,   # px in world frame
                             r * jnp.cos(theta),             # pz in world frame 
                             r_dot * jnp.sin(theta) + r * theta_dot * jnp.cos(theta),  # vx
                             r_dot * jnp.cos(theta) - r * theta_dot * jnp.sin(theta)]) # vz
    
    return x_cartesian

# Cartesian to Polar
def cartesian_to_polar(x_cartesian: np.array,
                       p_foot_W: np.array,
                       sys_params: slip_params) -> np.array:
    """
    Convert the cartesian coordinates to polar coordinates.
    """
    # COM cartesian state, x = [px, pz, vx, vz]
    px_com_W = x_cartesian[0]
    pz_com_W = x_cartesian[1]
    vx = x_cartesian[2]
    vz = x_cartesian[3]

    # get leg vector from the COM and foot positions
    px = px_com_W - p_foot_W[0]
    pz = pz_com_W - p_foot_W[1]

    # full state in polar coordinates
    r = jnp.sqrt(px**2 + pz**2)
    x_cartesian = jnp.array([r,                           # r
                             jnp.arctan2(px, pz),         # theta
                             (px * vx + pz * vz) / r,     # r_dot
                             (pz * vx - px * vz) / r**2]) # theta_dot

    return x_cartesian

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

    # flight fowrward propagation
    # x0 = jnp.array([0,     # px
    #                 3.0,   # pz
    #                 1.0,   # vx
    #                 0.5])  # vz

    # t_apex, t_impact, x_apex, x_impact = slip_flight_fwd_prop(x0, 0.0, sys_params)
    # print(f"t_apex: {t_apex}")
    # print(f"t_impact: {t_impact}")
    # print(f"x_apex: {x_apex}")
    # print(f"x_impact: {x_impact}")

    # ground forward propagation
    x0 = jnp.array([1.0,  # r
                    0.0,  # theta
                    0.0,  # r_dot
                    0.25]) # theta_dot
    res = slip_ground_fwd_prop(x0, sys_params)

    _, x_final, _, x_t, _ = res

    print(x_t)
    print(x_t.shape)

    # extract the non nan values
    x_final = x_t[~jnp.isnan(x_t).any(axis=1)]

    # convert all coordinates to cartesian
    for i in range(x_final.shape[0]):
        x_final = x_final.at[i, :].set(polar_to_cartesian(x_final[i], jnp.array([0.0, 0.0]), sys_params))
        
    # plot the results
    plt.plot(x_final[:,0], x_final[:,1])
    plt.show()

