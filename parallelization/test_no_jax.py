import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import time

def VanDerPol(t, x) -> np.ndarray:
    """
    Nonlinear dynamics of the Van der Pol oscillator
    """
    # Parameters
    mu = 10

    # Van der Pol oscillator
    xdot = np.array([
        x[1],
        mu * (1 - x[0]**2) * x[1] - x[0]
    ])

    return xdot

######################################################################################

if __name__ == "__main__":

    # Forward propagate dynamics
    dt = 0.05
    N = 100
    t0, tf = 0, dt * (N - 1)  # Final time step is adjusted by dt*N
    t_span = (t0, tf)
    t_eval = np.linspace(t0, tf, N)
    x0 = np.array([1.0, 1.0])

    # Solve ODE several times sequentially
    t_sum = 0.0
    num_sims = 100
    for i in range(num_sims):
        t0 = time.time()
        sol = solve_ivp(VanDerPol, t_span, x0, t_eval=t_eval, method="RK45")
        t1 = time.time()
        print("Elapsed time: ", t1 - t0)
        t_sum += t1 - t0
    print(f"Average time: {t_sum/num_sims}")
    print(f"Total time: {t_sum}")

    # Extract results
    x1_t = sol.y[0, :]
    x2_t = sol.y[1, :]

    # save the results into a csv file
    data = np.hstack((t_eval.reshape(-1, 1), x1_t.reshape(-1, 1), x2_t.reshape(-1, 1)))
    np.savetxt("./data/solution.csv", data, delimiter=",")
