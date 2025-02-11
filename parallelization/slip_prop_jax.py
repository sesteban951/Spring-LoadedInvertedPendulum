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

# simulation parameters data class
@struct.dataclass
class sim_params:
    dt_flight: float # sampling rate for flight phase
    dt_ground: float # sampling rate for ground phase
    N_flight: int    # max number of flight phase samples
    N_ground: int    # max number of ground phase samples

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
@partial(jit, static_argnums=(2,3))
def slip_flight_fwd_prop(x0: jnp.array, 
                         alpha: float,
                         params: slip_params,
                         sim_params: sim_params):
    '''
    Simulate the flight phase of the SLIP model
    '''
    # unpack the initial state
    px = x0[0]  # x-pos in world frame
    pz = x0[1]  # z-pos in world frame
    vx = x0[2]
    vz = x0[3]

    # compute the time until apex and impact, (vz(t) = vz_0 - g*t)
    t_apex = vz / params.g

    # compute the apex state
    x_apex = jnp.array([px + vx * t_apex,
                        pz + vz * t_apex - 0.5 * params.g * t_apex**2,
                        vx,
                        vz - params.g * t_apex])

    # compute the impact time,  (pz(t) = pz_0 + vz_0*t - 0.5*g*t^2)
    #                         => 0 = (pz_0 - pz_impact) + vz_0*t_impact - 0.5*g*t_impact^2
    pz_impact = params.l0 * jnp.cos(alpha)
    a = -0.5 * params.g
    b = vz
    c = pz - pz_impact
    s = jnp.sqrt(b**2 - 4*a*c)
    t1 = (-b + s) / (2*a)
    t2 = (-b - s) / (2*a)
    t_impact = jnp.maximum(t1, t2)

    # Compute the impact state
    x_impact = jnp.array([px + vx * t_impact,
                          pz + vz * t_impact - 0.5 * params.g * t_impact**2,
                          vx,
                          vz - params.g * t_impact])
    
    # compute the final foot position at impact
    p_foot = jnp.array([x_impact[0] - params.l0 * jnp.sin(alpha), 
                        x_impact[1] - params.l0 * jnp.cos(alpha)])

    # boolean function to stop the while loop
    def _condition_fun(args):

        # unpack the arguments
        _, _, _, impact = args

        return ~impact
    
    # main loop to simulate the flight phase
    def _compute_state(args):

        # unpack the arguments
        i, x_t, p_foot_t, impact = args

        # current time
        t = i * sim_params.dt_flight

        # update the state
        x_k = jnp.array([px + vx * t,
                         pz + vz * t - 0.5 * params.g * t**2,
                         vx,
                         vz - params.g * t])
        
        # update the foot position
        p_foot_k = jnp.array([x_k[0] - params.l0 * jnp.sin(alpha), 
                              x_k[1] - params.l0 * jnp.cos(alpha)])
        
        # insert the state into the array
        x_t = x_t.at[i].set(x_k)
        p_foot_t = p_foot_t.at[i].set(p_foot_k)

        # compute if we have reached the impact
        impact = (t >= t_impact)

        # increment the counter
        i += 1

        return (i, x_t, p_foot_t, impact)
    
    # loop to populate the arrays
    res = lax.while_loop(_condition_fun,
                         _compute_state,
                         (0, 
                         jnp.full((sim_params.N_flight,4), jnp.nan), 
                         jnp.full((sim_params.N_flight,2), jnp.nan), 
                         False))
    
    # unpack the results
    t_span = jnp.arange(0, sim_params.N_flight) * sim_params.dt_flight
    _, x_t, p_foot_t, _ = res

    return t_apex, x_apex, t_impact, x_impact, p_foot, t_span, x_t, p_foot_t

# SLIP ground dynamics
@ partial(jit, static_argnums=(2))
def slip_ground_dynamics(x: jnp.array,
                         u: float,
                         params: slip_params) -> jnp.array:
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
@partial(jit, static_argnums=(2,3))
def slip_ground_fwd_prop(x0: jnp.array,
                         p_foot: jnp.array,
                         params: slip_params,
                         sim_params: sim_params):
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
        i, x_prev, u_t, x_t, take_off = args

        # get the current control input
        u = u_t[i]

        # do the RK2 step        
        f1 = slip_ground_dynamics(x_prev, u, params)
        f2 = slip_ground_dynamics(x_prev + 0.5 * sim_params.dt_ground * f1, u, params)
        x_next = x_prev + sim_params.dt_ground * f2

        # print the counter
        i += 1

        # append to the history
        x_next_cart = polar_to_cartesian(x_next, p_foot)
        x_t = x_t.at[i].set(x_next_cart)

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

        return (i, x_next, u_t, x_t, take_off)

    # define a input signal, counting from 0 to max_iters
    res = lax.while_loop(_condition_fun,
                         _RK2_step,
                         (0,
                          x0,
                          jnp.zeros(sim_params.N_ground),
                          jnp.full((sim_params.N_ground,4), jnp.nan),
                          False))

    # things to return
    iters, x_takeoff, _, x_t, _ = res
    t_takeoff = iters * sim_params.dt_ground
    x_takeoff = polar_to_cartesian(x_takeoff, p_foot)
    t_span = jnp.arange(0, sim_params.N_ground) * sim_params.dt_ground
    x_t = x_t.at[0].set(polar_to_cartesian(x0, p_foot))

    return t_takeoff, x_takeoff, t_span, x_t

######################################################################################
# CONTROL
######################################################################################

# 

######################################################################################
# COORDINATE CONVERSION
######################################################################################

# Polar to Cartesian
@jit
def polar_to_cartesian(x_polar: np.array,
                       p_foot_W: np.array) -> np.array:
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
@jit
def cartesian_to_polar(x_cartesian: np.array,
                       p_foot_W: np.array) -> np.array:
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
    x_polar = jnp.array([r,                # r
              jnp.arctan2(px, pz),         # theta
              (px * vx + pz * vz) / r,     # r_dot
              (pz * vx - px * vz) / r**2]) # theta_dot

    return x_polar

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
    sys_param = slip_params(m  = float(1.0),     # mass [kg]
                            l0 = float(1.0),     # leg free length [m]
                            k  = float(500.0),   # leg stiffness [N/m]
                            br = float(0.0),     # radial damping [Ns/m]
                            ba = float(0.0),     # angular damping [Ns/m]
                            g  = float(9.81),    # gravity [m/s^2]
                            amax = float(1.5),   # max angle control input [rad]
                            amin = float(-1.5),  # min angle control input [rad]
                            umax = float(10.0),  # max control input [N]
                            umin = float(-10.0), # min control input [N]
                            )
    
    # Define the simulation parameters
    sim_param = sim_params(dt_flight = float(0.01),  # sampling rate for flight phase
                           dt_ground = float(0.005),  # sampling rate for ground phase
                           N_flight = int(150),       # number of flight phase samples
                           N_ground = int(250)        # number of ground phase samples
                           )

    ######################################################################################

    # flight fowrward propagation
    x0_cart = jnp.array([0,     # px
                         2.0,   # pz
                         0.0,   # vx
                         0.0])  # vz
    alpha = 15 * jnp.pi / 180

    N_sims = 2500
    T = np.zeros(N_sims)
    for i in range(N_sims):
        t0 = time.time()
        t_apex, x_apex, t_impact, x_impact, p_foot, t_span, x_t, p_foot_t = slip_flight_fwd_prop(x0_cart, alpha, sys_param, sim_param)
        t_apex.block_until_ready()
        x_apex.block_until_ready()
        t_impact.block_until_ready()
        x_impact.block_until_ready()
        p_foot.block_until_ready()
        t_span.block_until_ready()
        x_t.block_until_ready()
        p_foot_t.block_until_ready()

        x0_polar = cartesian_to_polar(x_impact, p_foot).block_until_ready()
        t_takeoff, x_takeoff, t_span, x_t = slip_ground_fwd_prop(x0_polar, p_foot, sys_param, sim_param)
        t_takeoff.block_until_ready()
        x_takeoff.block_until_ready()
        t_span.block_until_ready()
        x_t.block_until_ready()

        t_apex, x_apex, t_impact, x_impact, p_foot, t_span, x_t, p_foot_t = slip_flight_fwd_prop(x_takeoff, -alpha, sys_param, sim_param)
        t_apex.block_until_ready()
        x_apex.block_until_ready()
        t_impact.block_until_ready()
        x_impact.block_until_ready()
        p_foot.block_until_ready()
        t_span.block_until_ready()
        x_t.block_until_ready()
        p_foot_t.block_until_ready()

        tf = time.time()
        dt = tf - t0
        T[i] = dt
        print(f"Simulation time: {dt} seconds")

    print(f"Average simulation time: {np.mean(T)} seconds")
    print(f"Max simulation time: {np.max(T)} seconds")
    print(f"Min simulation time: {np.min(T)} seconds")
