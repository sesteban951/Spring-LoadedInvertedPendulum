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

######################################################################################
# DYNAMICS
######################################################################################

# SLIP ground dynamics (polar coordinates)
def slip_ground_dyn(xk: np.array,
                    uk: float,
                    params: slip_params):
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
        r * theta_dot**2 - params.g*np.cos(theta) + (params.k/params.m)*(params.l0 - r) - (params.br/params.m)*r_dot + (1/params.m)*uk,
        -(2/r) * r_dot*theta_dot + (params.g/r) * np.sin(theta) - (params.ba/params.m)*theta_dot
    ])

    return xdot

# SLIP ground dynamics (returns in polar local frame)
def slip_ground_fwd_prop(x0: np.array, 
                         dt: float, 
                         params:float):
    """
    Simulate the ground phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    
    # handle the crashing error
    crash = (abs(x0[1]) >= np.pi/2)
    if crash:
        print("The SLIP has fallen over in ground phase. Pole angle is: {:.2f} [deg]".format(x0[1] * 180/np.pi))
        return None, None, None, None

    # container for the trajectory, TODO: figure out a static size for the container. Dynamic = bad
    u_t = []
    x_t = []
    x_t.append(x0)

    # do Integration until you hit the switching manifold
    k = 0
    xk = x0
    dot_product, leg_uncompressed = False, False
    while not (leg_uncompressed * dot_product): # TODO: consider better switching detection
                                                # Min's suggestion: write the equation h(x)=0 that defines the switching surface and do root finding up to a tolerance

        # RK2 integration (in polar coordinates)
        uk = ground_control(xk, params)       
        f1 = slip_ground_dyn(xk, uk, params)
        f2 = slip_ground_dyn(xk + 0.5 * dt * f1, uk, params)
        xk = xk + dt * f2
        x_t.append(xk) #  TODO: figure out a static size for the container. Dynamic = bad
        u_t.append(uk) #  TODO: figure out a static size for the container. Dynamic = bad

        # compute take-off guard conditions
        x_cart = polar_to_cartesian(xk, np.array([0, 0]), params)  # consider just saving this vector directly to save on compute
        leg_pos = np.array([x_cart[0], x_cart[1]])
        leg_vel = np.array([x_cart[2], x_cart[3]])
        dot_product = (np.dot(leg_pos, leg_vel) >= 0)
        leg_uncompressed = (xk[0] >= params.l0)

        # increment the counter
        k += 1

    # convert the trajectory to a numpy array
    x_t = np.array(x_t)
    u_t.append(uk)      # the last point is the same as the previous (to maintain same size as state)
    u_t = np.array(u_t)
    u_t = u_t.reshape(-1, 1)

    # domain information (1 for ground phase)
    N = len(x_t)
    D_t = np.ones((N, 1))

    # time span infomration
    t_span = np.linspace(0, (N-1)*dt, N)
    t_span = t_span.reshape(-1, 1)

    return t_span, x_t, u_t, D_t

# SLIP flight dynamics (returns in cartesian world frame)
def slip_flight_fwd_prop(x0: np.array, 
                         dt: float,
                         apex_terminate: bool,
                         params: slip_params):
    """
    Simulate the flight phase of the Spring Loaded Inverted Pendulum (SLIP) model.
    """
    # TODO: Consider adding a velocity input in the leg for flight phase.

    # unpack the initial state
    px_0 = x0[0]  # these are in world frame
    pz_0 = x0[1]  # these are in world frame
    vx_0 = x0[2]
    vz_0 = x0[3]

    # handle crashing error
    crash = (pz_0 <= 0)
    if crash:
        print("The SLIP has fallen over in flight phase. COM height is: {:.2f} [m]".format(pz_0))
        return None, None, None, None, None, None, None

    # compute a control action for the impact angle
    v_des = 0.0
    alpha = angle_control(x0, v_des, params)

    # compute time until apex
    if vz_0 >= 0:
        t_apex = vz_0 / params.g  # from vz(t) = vz_0 - g*t
    else:
        t_apex = None

    # Apex Condition: if you want to terminate at the apex
    if apex_terminate is True:
        
        # find the zero velocity time
        no_apex = vz_0 < 0
        if no_apex:
            print("The SLIP has no apex time. The z-velocity is not upwards. vz_0 = {:.2f} [m/s]".format(vz_0))
            print(alpha)
            return None, None, None, None, None, None, None
        else:
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

        # handle when there is no feasible apex time
        no_impact_time = t_terminate < 0
        if no_impact_time:
            print("The SLIP has no impact time. The time until impact is negative. t_impact = {:.2f} [s]".format(t_terminate))
            return None, None, None, None, None, None, None

    # create a time vector
    t_span = np.arange(0, t_terminate, dt)
    t_span = np.append(t_span, t_terminate)
    t_span = t_span.reshape(-1, 1)

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
    for i in range(len(t_span)):

        # update the state
        t = t_span[i, 0]
        x_t[i, 0] = px_0 + vx_0 * t                          # pos x
        x_t[i, 1] = pz_0 + vz_0 * t - 0.5 * params.g * t**2  # pos z
        x_t[i, 2] = vx_0                                     # vel x       
        x_t[i, 3] = vz_0 - params.g * t                      # vel z

        # check that the COM is above the ground
        assert pz_0 > 0, "The center of mass is under ground. pz = ".format(pz_0)

    # generate a control input vector (should be zeros since no control in flight phase)
    u_t = np.zeros((len(t_span),1))

    # Domain information (0 for flight phase)
    D_t = np.zeros((len(t_span), 1))

    # compute the final leg position in world frame
    x_com = x_t[-1, :]     # take the last state as the final state
    p_foot = np.array([x_com[0] - params.l0 * np.sin(alpha),
                       x_com[1] - params.l0 * np.cos(alpha)])

    return t_span, x_t,  x_apex, u_t, alpha, D_t, p_foot

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
    X_state = []  # cartesian state
    X_apex  = []  # apex states
    U_ground = [] # radial inputs
    U_flight = [] # flight landing angle
    T = []  # time
    D = []  # domain
    P = []  # foot position

    # forward propogate the SLIP model for N_apex discrete steps
    apex_terminate = False
    t_current = 0.0

    t_span, x_t, x_apex, u_t, alpha, D_t, p_foot = slip_flight_fwd_prop(x0, dt, apex_terminate, params)
    if t_span is None:
        print("SLIP failure in flight phase.")
        return None, None, None, None, None, None, None

    t_span = t_span + t_current
    T.append(t_span)
    X_state.append(x_t)
    U_ground.append(u_t)
    U_flight.append(alpha)
    X_apex.append(x_apex)
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
        t_span, x_t, u_t, D_t = slip_ground_fwd_prop(x0_polar, dt, params)
        for i in range(len(x_t)):
            x_t[i, :] = polar_to_cartesian(x_t[i, :], p_foot, params) # convert polar to cartesian
        t_span = t_span + t_current
        
        X_state.append(x_t)
        U_ground.append(u_t)
        T.append(t_span)
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
        t_span, x_t, x_apex, u_t, alpha, D_t, p_foot = slip_flight_fwd_prop(x0, dt, apex_terminate, params)
        if t_span is None:
            print("SLIP failure in flight phase.")
            return None, None, None, None, None, None, None

        t_span = t_span + t_current
        X_state.append(x_t)
        X_apex.append(x_apex)
        U_ground.append(u_t)
        U_flight.append(alpha)
        T.append(t_span)
        D.append(D_t)
        P.append(p_foot)
        t_current = t_span[-1]

    return T, X_state, X_apex, U_flight, U_ground, P, D

######################################################################################
# CONTROL
######################################################################################

# simple leg landing controller
def angle_control(x_flight: np.array,
                  v_des: float,
                  params: slip_params) -> float:
    """
    Simple Raibert controller for the SLIP model.
    """
    # unpack the state
    px = x_flight[0]
    pz = x_flight[1]
    vx = x_flight[2]
    vz = x_flight[3]

    # compute the desired angle from simple raibert controller
    # kd = 0.13
    # alpha = -kd * (vx - v_des) 

    # sample from normal distribution
    mu = 0.0
    sigma = 0.5
    alpha = np.random.normal(mu, sigma)

    # saturate the control input
    alpha = np.clip(alpha, params.amin, params.amax)

    return alpha

# TODO: define some control for the prismatic joint in ground phase
def ground_control(xk: np.array,
                   params: slip_params) -> float:
    """
    Controller for the prismatic joint in the SLIP model during ground phase.
    """

    # TODO: eventually put something here
    u = 0.0

    # saturate the control input
    if (u > params.umax) or (u < params.umin):
        print("Prismatic control input is out of bounds. Control input is: {:.2f} [N]".format(u))
        u = np.clip(u, params.umin, params.umax)

    return u

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
    sys_params = slip_params(m  = 1.0,    # mass [kg]
                             l0 = 1.0,    # leg free length [m]
                             k  = 1000.0,  # leg stiffness [N/m]
                             br = 0.0,    # radialdamping [Ns/m]
                             ba = 0.0,    # angular damping [Ns/m]
                             g  = 9.81,   # gravity [m/s^2]
                             amax = 1.2,  # max angle control input [rad]
                             amin = -1.2, # min angle control input [rad]
                             umax = 10.0, # max control input [N]
                             umin = -10.0 # min control input [N]
                            )

    # simulation parameters
    dt = 0.001
    apex_terminate = False

    ###########################################################################

    t0 = time.time()

    # simulate the SLIP model
    x0_cart_W = np.array([0.0,  
                          2.0,  
                          0.0,  
                          0.0]) 
    x_apex_des = np.array([0.0,  # pz 
                           1.0]) # vx
    
    # simulation parameters
    K = 5000         # number of rollouts to simulate
    counter = 0      # counter for successful simulations
    counter_tot = 0  # counter for total simulations
    N_apex = 1       # choose which apex to stop at

    # forward rollout until you get K successful simulations
    sol_list = []
    while counter < K:

        # do a forward rollout
        T, X_state, X_apex, U_flight, U_ground, P, D = slip_prop(x0_cart_W, dt, N_apex, sys_params)

        # check if the simulation was successful
        if T is None:
            pass
        else:
            sol = (T_, X_state_, X_apex_, U_flight_, U_ground_, P_, D_) = (T, X_state, X_apex, U_flight, U_ground, P, D)
            sol_list.append(sol)
            counter += 1

        # increment the total counter
        counter_tot += 1

    # check which one is the closest to the desired apex state
    vx_list = []
    alpha_list = []
    for i in range(K):
        
        # extract the solution of interest
        sol_i = sol_list[i]
        
        # extract the x-velocity
        X_apex_i = sol_i[2]
        x_apex_i_last = X_apex_i[-1]
        vx_i = x_apex_i_last[2]
        vx_list.append(vx_i)

        # extract the landing angle
        U_flight_i = sol_i[3]
        alpha_i = U_flight_i[0]
        alpha_list.append(alpha_i)

    vx_list = np.array(vx_list)
    alpha_list = np.array(alpha_list)

    # find best rollout
    idx = np.argmin(np.abs(vx_list - x_apex_des[1]))
    T, X_state, X_apex, U_flight, U_ground, P, D = sol_list[idx]

    tf = time.time()

    print("*" * 80)
    print("The best simulation is: ", idx)
    print("The desired apex state is: ", x_apex_des)
    print("The actual apex state is: ", X_apex[-1])
    print("Simulation time: {:.2f} [s]".format(tf - t0))
    print("Total number of simulations: ", counter_tot)
    print("Number of successful simulations: ", counter)
    print("Percentage of successful simulations: {:.2f} [%]".format(counter / counter_tot * 100))

    ###########################################################################

    # plot a histogram of the x-velocity
    plt.figure()
    plt.hist(vx_list, bins=100)
    plt.xlabel('vx [m/s]')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    # plot a histogram of the landing angle
    plt.figure()
    plt.hist(alpha_list, bins=100)
    plt.xlabel('alpha [rad]')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    # plot the states
    plt.figure()
    for i in range(len(T)):
        plt.plot(X_state[i]
        [:, 0], X_state[i][:, 1])

    # plot the apex states
    for i in range(len(X_apex)):
        if X_apex[i] is not None:
            
            plt.plot(X_apex[i][0], X_apex[i][1], 'rx')

            # plot an arrow to show the velocity vector
            vel_scaling = 0.5
            plt.arrow(X_apex[i][0], X_apex[i][1], 
                      X_apex[i][2] * vel_scaling, X_apex[i][3] * vel_scaling, 
                      head_width=0.05, head_length=0.01, fc='k', ec='k')

    # plot the foot position
    for i in range(len(P)):
        plt.plot(P[i][0], P[i][1], 'ro')

    plt.xlabel('px [m]')
    plt.ylabel('pz [m]')
    plt.grid()
    plt.show()
