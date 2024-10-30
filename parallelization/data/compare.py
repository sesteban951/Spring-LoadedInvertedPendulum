import numpy as np
import matplotlib.pyplot as plt

# import the data
sol_jax = np.loadtxt("./data/solution_jax.csv", delimiter=",")
sol_jax_batched = np.loadtxt("./data/solution_jax_batch.csv", delimiter=",")
sol_no_jax = np.loadtxt("./data/solution.csv", delimiter=",")

# plot the results
plt.plot(sol_jax[:, 1], sol_jax[:, 2], label="JAX", linewidth=0.5, color='green')
plt.plot(sol_no_jax[:, 1], sol_no_jax[:, 2], label="No JAX", linewidth=0.5, color='red')
plt.plot(sol_jax_batched[:, 1], sol_jax_batched[:, 2], label="JAX Batched", linewidth=0.5, color='blue')
plt.plot(sol_jax[0, 1], sol_jax[0, 2], 'go')
plt.plot(sol_jax[-1, 1], sol_jax[-1, 2], 'gx')
plt.plot(sol_jax_batched[0, 1], sol_jax_batched[0, 2], 'bo')
plt.plot(sol_jax_batched[-1, 1], sol_jax_batched[-1, 2], 'bx')
plt.plot(sol_no_jax[0, 1], sol_no_jax[0, 2], 'ro')
plt.plot(sol_no_jax[-1, 1], sol_no_jax[-1, 2], 'rx')

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Van der Pol Oscillator")
plt.axis('equal')  # Set the axis to be equal
plt.show()