import numpy as np

import jax
from jax import lax
from jax.lib import xla_bridge
from jax import jit, vmap
import jax.numpy as jnp

from functools import partial
import time

######################################################################################

# TODO: Consider making a data class to hold everything
# @dataclass
# class SomeDataClass:
#     """
#     Data class to hold some data
#     """
#     def __init__(self, x: int, y: int) -> None:
#         self.x = x
#         self.y = y

#     def __repr__(self) -> str:
#         return f"x: {self.x}, y: {self.y}"

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

# simple jax test
def test_jax() -> float:
    """
    Simple test to make sure that JAX is working
    """
    # random seed
    seed = 0

    # create a random array
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    x1 = jax.random.uniform(key, (20,), minval=0.0, maxval=1.0)
    x2 = jax.random.uniform(subkey, (20,), minval=0.0, maxval=1.0)

    # compute the mean
    mean = jnp.mean(x2)

    return mean

######################################################################################

# Van der Pol oscillator dynamics to test things out
@jit # TODO: could get rid of this jit and then run just as fast. 
def VanDerPol_jit(t,x) -> jnp.ndarray:
    """
    Nonlinear dynamics of the Van der Pol oscillator
    """
    # Parameters
    mu = 10

    # Van der Pol oscillator
    xdot = jnp.array([
        x[1],
        mu * (1 - x[0]**2) * x[1] - x[0]
    ])

    return xdot

# Forward propagate dynamics
@partial(jit, static_argnums=(1, 2))
def fwd_propagate_dyn_jit(x0, dt, N) -> jnp.ndarray:

    # RK2 integration | Source: https://www.youtube.com/watch?v=HOWJp8NV5xU&t=1087s
    def _rk2_step(carry, i):
        t_current = i * dt
        x_prev = carry
        f1 = VanDerPol_jit(t_current, x_prev)
        f2 = VanDerPol_jit(t_current + dt * 0.5, x_prev + 0.5 * dt * f1)
        x_next = x_prev + dt * f2
        return x_next, x_next

    # Use lax.scan to apply the RK2 steps iteratively
    _, x = lax.scan(_rk2_step, x0, jnp.arange(1, N))
    
    # Stack initial state with the results from lax.scan for complete trajectory
    x = jnp.vstack([x0, x])

    return x

######################################################################################

if __name__ == "__main__":

    # print info
    print_platform_info()

    # do some basic computation to make sure JAX is working
    mean = test_jax()

    # Forward propagate dynamics
    dt = 0.05
    N = 100
    x0 = jnp.array([1.0, 1.0])

    # TODO: investigate why I need a warm start to get to a fast speed
    sol = fwd_propagate_dyn_jit(x0, dt, N).block_until_ready()

    # solve the ODE several times sequentially
    t_sum = 0.0
    num_sims = 100
    for i in range(num_sims):    
        t0 = time.time()
        sol = fwd_propagate_dyn_jit(x0, dt, N).block_until_ready() # TODO: can get rid of block_until_ready()
                                                                   #     only use this when you must need to wait  
        t1 = time.time()
        t_sum += t1-t0
    print(f"Average time (sequentially): {t_sum/num_sims}")
    print(f"Total time (sequentially): {t_sum}")

    # Parallel simulations
    x0_batch = jnp.array([[1.0, 1.0]] * num_sims)  # Ensure x0_batch is shaped correctly
    fwd_propagate_dyn_vmap = vmap(fwd_propagate_dyn_jit, in_axes=(0, None, None))

    # TODO: investigate why I need a warm start to get to a fast speed
    sol_batch = fwd_propagate_dyn_vmap(x0_batch, dt, N).block_until_ready()

    t0 = time.time()
    sol_batch = fwd_propagate_dyn_vmap(x0_batch, dt, N).block_until_ready()
    t1 = time.time()
    print(f"Average time (parallel): {(t1 - t0) / num_sims}")
    print(f"Total time (parallel): {t1 - t0}")

    # retreive the jax array as a numpy array
    sol = np.array(sol, copy=False) # TODO: transferring from GPU to CPU is expensive. Try doing all in jax data types
    data = np.hstack((np.linspace(0, dt*(N-1), N).reshape(-1, 1), sol))

    # retreive the last batched solution
    sol_batch = np.array(sol_batch, copy=False)
    data_batch = np.hstack((np.linspace(0, dt*(N-1), N).reshape(-1, 1), sol_batch[0]))

    # Save the solution into a CSV file for plotting later  
    np.savetxt("./data/solution_jax.csv", data, delimiter=",")
    np.savetxt("./data/solution_jax_batch.csv", data_batch, delimiter=",")
