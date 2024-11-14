
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
@partial(jit, static_argnums=(2, 3))
def slip_flight_fwd_prop(x0: np.array, 
                         alpha: float, 
                         dt: float, 
                         sys_params: slip_params):
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
        return x * (-np.inf)
    def pass_through(x):
        return x
    res = lax.cond(vz < 0, raise_error, pass_through, vz) 

    # compute the time until apex and impact
    t_apex = vz / sys_params.g

    # compute the impact time
    pz_impact = sys_params.l0 * jnp.cos(alpha)
    
    return t_apex
    
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
    x0 = jnp.array([0,  # px
                    3.0,  # pz
                    1.0,  # vx
                    -0.5]) # vz

    a = slip_flight_fwd_prop(x0, 0.0, sys_params.dt, sys_params)

