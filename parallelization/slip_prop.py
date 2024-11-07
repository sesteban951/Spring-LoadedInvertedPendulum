import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
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

######################################################################################
# DYNAMICS
######################################################################################

# SLIP ground dynamics (polar coordinates)
def slip_ground_dyn(xk: np.array, 
                    params: slip_params) -> np.array:
    """
    Closed form dynamics for the ground phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # polar state, x = [r, theta, rdot, thetadot]
    r = xk[0]
    theta = xk[1]
    r_dot = xk[2]
    theta_dot = xk[3]

    # ground phase dynamics
    xdot = np.array([
        r_dot,
        theta_dot,
        r * theta_dot**2 - params.g * np.cos(theta) + (params.k/params.m) * (params.l0 - r),
        -(2/r) * r_dot*theta_dot + (params.g/r) * np.sin(theta)
    ])

    return xdot

# SLIP ground dynamics (returns in polar local frame)
def slip_ground_fwd_prop(x0: np.array, 
                         dt: float, 
                         params:float) -> np.array:
    """
    Simulate the ground phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # ensure the SLIP hasn't fallen over
    assert (abs(x0[1]) <= np.pi/2), "The SLIP has fallen over in ground phase. Pole angle is: {:.2f} [deg]".format(x0[1] * 180/np.pi)

    # container for the trajectory, TODO: figure out a static size for the container. Dynamic = bad
    x_t = []
    x_t.append(x0)    

    # do Integration until you hit the switching manifold
    k = 0
    xk = x0
    dot_product, leg_uncompressed = False, False
    while not (leg_uncompressed * dot_product): # TODO: consider better switching detection

        # RK2 integration (in polar coordinates)
        f1 = slip_ground_dyn(xk, params)
        f2 = slip_ground_dyn(xk + 0.5 * dt * f1, params)
        xk = xk + dt * f2
        x_t.append(xk) #  TODO: figure out a static size for the container. Dynamic = bad

        # check take-off guard conditions
        x_cart = polar_to_cartesian(xk, np.array([0, 0]), params)  # consider just saving this vector directly to save on compute
        leg_pos = np.array([x_cart[0], x_cart[1]])
        leg_vel = np.array([x_cart[2], x_cart[3]])
        dot_product = (np.dot(leg_pos, leg_vel) >= 0)
        leg_uncompressed = (xk[0] >= params.l0)

        # increment the counter
        k += 1

    # convert the trajectory to a numpy array
    N = len(x_t)
    x_t = np.array(x_t)

    # domain information (1 for ground phase)
    D_t = np.ones((N, 1))

    # time span infomration
    t_span = np.linspace(0, (N-1)*dt, N)

    return t_span, x_t, D_t

# SLIP flight dynamics (returns in cartesian world frame)
def slip_flight_fwd_prop(x0: np.array, 
                         dt: float, 
                         apex_terminate: bool,
                         params: slip_params) -> np.array:
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
    assert pz_0 > 0, "The center of mass is under ground. pz = ".format(pz_0)

    # choose control action
    vx_des = 0.0
    alpha = raibert_controller(x0, vx_des, params)

    # compute time until apex
    if vz_0 >= 0:
        t_apex = vz_0 / params.g  # from vz(t) = vz_0 - g*t
    else:
        t_apex = None

    # Apex Condition: if you want to terminate at the apex
    if apex_terminate is True:
        # find the zero velocity time
        assert vz_0 >= 0, "The SLIP z-velocity is not upwwards, therfore no such positive apex time exists."
        t_terminate = t_apex

    # Guard Condition: compute the time until impact
    else:
        pz_impact = params.l0 * np.cos(alpha)
        a = 0.5 * params.g
        b = - vz_0
        c = -(pz_0 - pz_impact)
        s = np.sqrt(b**2 - 4*a*c)
        d = 2 * a
        t1 = (-b + s) / d
        t2 = (-b - s) / d
        t_terminate = max(t1, t2)
        assert t_terminate >= 0, "Time until impact must be positive or equal to zero."

    # create a time vector
    t_span = np.arange(0, t_terminate, dt)
    t_span = np.append(t_span, t_terminate)

    # create a trajectory vector
    x_t = np.zeros((len(t_span), 4))

    # compute the apex state
    # there exists feasible apex time
    if t_apex is not None:   
        x_apex = np.array([px_0 + vx_0 * t_apex,
                           pz_0 + vz_0 * t_apex - 0.5 * params.g * t_apex**2,
                           vx_0,
                           vz_0 - params.g * t_apex])
    # there does not exist feasible apex time
    else :                   
        x_apex = np.array([0, 0, 0, 0])

    # simulate the flight phase
    for i, t in enumerate(t_span):

        # update the state
        x_t[i, 0] = px_0 + vx_0 * t                          # pos x
        x_t[i, 1] = pz_0 + vz_0 * t - 0.5 * params.g * t**2  # pos z
        x_t[i, 2] = vx_0                                     # vel x       
        x_t[i, 3] = vz_0 - params.g * t                      # vel z

        # check that the COM is above the ground
        assert pz_0 > 0, "The center of mass is under ground. pz = ".format(pz_0)

    # Domain information (0 for flight phase)
    D_t = np.zeros((len(t_span), 1))

    # compute the final leg position in world frame
    x_com = x_t[-1, :]     # take the last state as the final state
    p_foot = np.array([x_com[0] - params.l0 * np.sin(alpha),
                       x_com[1] - params.l0 * np.cos(alpha)])

    return t_span, x_t, x_apex, D_t, p_foot

# SLIP full propogatoin (returns in cartesian world frame)
def slip_prop(x0: np.array,
              dt: float,
              N_apex: int,
              params: slip_params):
    """
    Propogate the Spring Loaded Inverted Pendulum (SLIP) model. Over n apex steps.
    Starts in the flight phase.
    """
    # check that you want at least one apex step
    assert N_apex >= 0, "The number of apex steps must be greater than or equal to zero."

    # container for the trajectory
    # TODO: figure out a static size for the container. Dynamic = bad
    X = []  # cartesian state
    T = []  # time
    D = []  # domain
    P = []  # foot position
    A = []  # apex states

    # forward propogate the SLIP model for N_apex discrete steps
    apex_terminate = False
    t_current = 0.0

    # do the first flight phase
    t_span, x_t, x_apex, D_t, p_foot = slip_flight_fwd_prop(x0, dt, apex_terminate, params)
    
    t_span = t_span + t_current
    T.append(t_span)
    X.append(x_t)
    A.append(x_apex)
    D.append(D_t)
    P.append(p_foot)
    t_current = t_span[-1]

    # while not reached the desired number of apex steps
    # TODO: might be double counting -- the first ground state and the last flight state, vice versa
    for k in range(N_apex):

        # set intial condition for ground phase
        xf_cart = x_t[-1, :]
        x0_polar = carteisan_to_polar(xf_cart, p_foot, params)

        # ground phase
        t_span, x_t, D_t = slip_ground_fwd_prop(x0_polar, dt, params)
        for i in range(len(x_t)):
            x_t[i, :] = polar_to_cartesian(x_t[i, :], p_foot, params) # convert polar to cartesian
        t_span = t_span + t_current
        
        T.append(t_span)
        X.append(x_t)
        D.append(D_t)
        t_current = t_span[-1]

        # update intial condition for flight phase
        x0 = x_t[-1, :]

        # if it's the last apex step, terminate at apex
        if k == N_apex - 1:
            apex_terminate = True
        else:
            apex_terminate = False

        # flight phase
        t_span, x_t, x_apex, D_t, p_foot = slip_flight_fwd_prop(x0, dt, apex_terminate, params)

        t_span = t_span + t_current
        T.append(t_span)
        X.append(x_t)
        A.append(x_apex)
        D.append(D_t)
        P.append(p_foot)
        t_current = t_span[-1]

    return T, X, A, P, D

######################################################################################
# CONTROL
######################################################################################

# simple raibert controller # TODO: this will be a normal distirbution
def raibert_controller(x_flight: np.array,
                       v_des: float,
                       params: slip_params) -> float:
    """
    Simple Raibert controller for the SLIP model.
    """
    # unpack the state
    vx = x_flight[2]

    # compute the desired angle
    kd = 0.13
    alpha = -kd * (vx - v_des) 

    return alpha

######################################################################################
# COORDINATE TRANSFORMATIONS
######################################################################################

# Polar to Cartesian coordiante (returns in world frame)
def polar_to_cartesian(x_polar: np.array, 
                       p_foot_W: np.array,
                       params: slip_params) -> np.array:
    """
    Convert the polar coordinates to cartesian coordinates.
    Assumes the foot is at (0,0), must add the last leg pos after this method to keep track in world frame
    """
    # polar state, x = [r, theta, rdot, thetadot]
    r = x_polar[0]
    theta = x_polar[1]
    r_dot = x_polar[2]
    theta_dot = x_polar[3]

    # full state in cartesian coordintes
    px = r * np.sin(theta)      # COM position x
    pz = r * np.cos(theta)      # COM position z
    vx = r_dot * np.sin(theta) + r * theta_dot * np.cos(theta)  # COM velocity x
    vz = r_dot * np.cos(theta) - r * theta_dot * np.sin(theta)  # COM velocity z

    # shift the COM position by the foot position
    px += p_foot_W[0]

    return np.array([px, pz, vx, vz])

# Cartesian to polar coordinate (returns in local polar frame)
def carteisan_to_polar(x_cart_W: np.array, 
                       p_foot_W : np.array,
                       params: slip_params) -> np.array:
    """
    Convert the cartesian coordinates to polar coordinates.
    Assumes the cartesian state and foot position are in world frame.
    """
    # flight state, x = [x, z, xdot, zdot]
    px_W = x_cart_W[0]
    pz_W = x_cart_W[1]
    vx = x_cart_W[2]
    vz = x_cart_W[3]

    # convert the world frame coordaintes to local frame
    px = px_W - p_foot_W[0]
    pz = pz_W - p_foot_W[1]

    # full state in polar coordinates
    r = np.sqrt(px**2 + pz**2)           # leg length
    th = np.arctan2(px, pz)              # leg angle
    r_dot = (px * vx + pz * vz) / r      # leg length rate
    th_dot = (vx * pz - px * vz) / r**2  # leg angle rate

    return np.array([r, th, r_dot, th_dot])

######################################################################################
# MAIN
######################################################################################

if __name__ == "__main__":

    # define the sytem parameters
    t0 = time.time()
    sys_params = slip_params(m  = 1.0,   # mass [kg]
                             l0 = 1.0,   # leg free length [m]
                             k  = 500.0,  # leg stiffness [N/m]
                             g  = 9.81)  # gravity [m/s^2]

    # simulation parameters
    dt = 0.005
    apex_terminate = False

    ###########################################################################

    x0_cart_W = np.array([0.0,  
                          2.0,  
                          1.0,  
                          0.0]) 
    N_apex = 3
    T, X, A, P, D = slip_prop(x0_cart_W, dt, N_apex, sys_params)

    print(len(T))
    print(len(X))
    print(len(A))
    print(len(P))
    print(len(D))

    tf = time.time()
    print("Time to run: ", tf - t0)

    # convert to CSV for saving
    T_data = np.hstack(T)
    X_data = np.vstack(X)
    A_data = np.vstack(A)
    P_data = np.vstack(P)
    D_data = np.vstack(D)

    # save the data
    np.savetxt("./data/slip_prop_T.csv", T_data, delimiter=",")
    np.savetxt("./data/slip_prop_X.csv", X_data, delimiter=",")
    np.savetxt("./data/slip_prop_A.csv", A_data, delimiter=",")
    np.savetxt("./data/slip_prop_P.csv", P_data, delimiter=",")
    np.savetxt("./data/slip_prop_D.csv", D_data, delimiter=",")

    # plot the states
    plt.figure()
    for i in range(len(T)):
        plt.plot(X[i]
        [:, 0], X[i][:, 1])

    # plot the apex states
    for i in range(len(A)):
        if A[i] is not None:
            plt.plot(A[i][0], A[i][1], 'rx')

    # plot the foot position
    for i in range(len(P)):
        plt.plot(P[i][0], P[i][1], 'ro')

    plt.xlabel('px [m]')
    plt.ylabel('pz [m]')
    plt.grid()
    plt.show()

    # plot the apex states
    plt.figure()
    for i in range(len(A)):
        if A[i] is not None:
            plt.plot(A[i][2], A[i][1], 'rx')
    plt.plot(A[0][2], A[0][1], 'go')
    plt.plot(A[-1][2], A[-1][1], 'kx')
    plt.xlabel('vx [m/s]')
    plt.ylabel('pz [m]')
    plt.grid()
    plt.show()
