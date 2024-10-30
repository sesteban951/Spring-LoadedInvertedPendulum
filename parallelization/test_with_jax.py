import numpy as np

import jax
from jax.lib import xla_bridge
from jax import jit, vmap
import jax.numpy as jnp
import jax.scipy as jsp

import time
import os

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
def VanDerPol(t,x) -> jnp.ndarray:
    """
    Nonlinear dynamics of the Van der Pol oscillator
    """
    # Parameters
    mu = 1.0

    # Van der Pol oscillator
    xdot = jnp.array([
        x[1],
        mu * (1 - x[0]**2) * x[1] - x[0]
    ])

    return xdot

# Forward propagate dynamics
def fwd_propagate_dyn(x0, dt, N) -> jnp.ndarray:

   # Initialize the state array
    x = jnp.zeros((N, len(x0)))  # Create an array to hold the states
    x = x.at[0].set(x0)  # Set the initial condition

    # Time stepping
    for i in range(1, N):
        t_current = i * dt
        # Compute the current state using the trapezoidal rule
        k1 = VanDerPol(t_current, x[i-1])
        k2 = VanDerPol(t_current + dt, x[i-1] + dt * k1)
        x = x.at[i].set(x[i-1] + dt * (k1 + k2) / 2)  # Trapezoidal update

    return x

######################################################################################

if __name__ == "__main__":

    # print info
    print_platform_info()

    # do some basic computation to make sure JAX is working
    mean = test_jax()

    # Forward propagate dynamics
    dt = 0.05
    N = 500
    x0 = jnp.array([1.0, 1.0])
    sol = fwd_propagate_dyn(x0, dt, N)

    # retreive the jax array as a numpy array
    sol = np.array(sol, copy=False)

    # Save the solution into a CSV file for plotting later

    csv_file_path = os.path.join("", "sol.csv")  # Use os.path.join for better compatibility
    np.savetxt(csv_file_path, jax.device_get(sol), delimiter=",")

