import numpy as np
import jax
from jax import lax
from jax.lib import xla_bridge
from jax import jit, vmap
import jax.numpy as jnp

from functools import partial
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time

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
