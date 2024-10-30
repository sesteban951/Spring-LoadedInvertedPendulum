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
    mu = 1.0

    # Van der Pol oscillator
    xdot = np.array([
        x[1],
        mu * (1 - x[0]**2) * x[1] - x[0]
    ])

    return xdot

if __name__ == "__main__":

    # Forward propagate dynamics
    dt = 0.05
    N = 500
    t0, tf = 0, dt * (N - 1)  # Final time step is adjusted by dt*N
    t_span = (t0, tf)
    t_eval = np.linspace(t0, tf, N)
    x0 = np.array([1.0, 1.0])

    # Solve ODE
    # t1 = time.time()
    sol = solve_ivp(VanDerPol, t_span, x0, t_eval=t_eval)
    # t2 = time.time()
    # print(f"Time to compute: {t2 - t1:.4f} seconds")

    # Extract results
    x1_t = sol.y[0, :]
    x2_t = sol.y[1, :]

    # Plot results over time
    plt.plot(x1_t, x2_t)
    plt.plot(x1_t[0], x2_t[0], 'ro')
    plt.plot(x1_t[-1], x2_t[-1], 'rx')
    plt.xlabel("Time")
    plt.ylabel("States")
    plt.title("Van der Pol Oscillator Dynamics")
    plt.show()
