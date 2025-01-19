#include "../inc/dynamics.h"


Dynamics::Dynamics(YAML::Node config_file)
{
    // set the system parameters
    this->params.m = config_file["SYS_PARAMS"]["m"].as<double>();
    this->params.g = config_file["SYS_PARAMS"]["g"].as<double>();
    this->params.k = config_file["SYS_PARAMS"]["k"].as<double>();
    this->params.b = config_file["SYS_PARAMS"]["b"].as<double>();
    this->params.l0 = config_file["SYS_PARAMS"]["l0"].as<double>();
    this->params.r_min = config_file["SYS_PARAMS"]["r_min"].as<double>();
    this->params.r_max = config_file["SYS_PARAMS"]["r_max"].as<double>();
    this->params.theta_min = config_file["SYS_PARAMS"]["theta_min"].as<double>();
    this->params.theta_max = config_file["SYS_PARAMS"]["theta_max"].as<double>();
    this->params.rdot_lim = config_file["SYS_PARAMS"]["rdot_lim"].as<double>();
    this->params.thetadot_lim = config_file["SYS_PARAMS"]["thetadot_lim"].as<double>();
    this->params.torque_ankle = config_file["SYS_PARAMS"]["torque_ankle"].as<bool>();
    this->params.torque_ankle_lim = config_file["SYS_PARAMS"]["torque_ankle_lim"].as<double>();
    this->params.torque_ankle_kp = config_file["SYS_PARAMS"]["torque_ankle_kp"].as<double>();
    this->params.torque_ankle_kd = config_file["SYS_PARAMS"]["torque_ankle_kd"].as<double>();
    this->params.interp = config_file["SYS_PARAMS"]["interp"].as<char>();
}


// NonLinear Dynamics function, xdot = f(x, u, d)
Vector_6d Dynamics::dynamics(Vector_6d x, Vector_2d u, Vector_2d p_foot, Domain d)
{
    // access some system parameters
    double m = this->params.m;
    double g = this->params.g;
    double k = this->params.k;
    double b = this->params.b;

    // unpack the state vector
    Vector_2d p_com, v_com;
    p_com << x(0), x(1);
    v_com << x(2), x(3);

    // vectors to use in the calculations
    Vector_2d a_com, v_leg, g_vec, f_com;
    Vector_6d xdot;
    g_vec << 0, -g;

    // Flight domain dynamics (F)
    if (d == Domain::FLIGHT) {
        // compute the flight dynamics
        a_com = g_vec;
        v_leg << u(0), u(1);

        // stack the acceleration and velocity vectors
        xdot << v_com, a_com, v_leg;
    }

    // Ground domain dynamics (G)
    else if (d == Domain::GROUND) {

        // compute the leg state
        Vector_2d r_vec = p_foot - p_com;
        double r_norm = r_vec.norm();
        Vector_2d r_hat = r_vec / r_norm;
        double r_x = r_vec(0);
        double r_z = r_vec(1);

        Vector_2d rdot_vec = -v_com;
        double rdot_x = rdot_vec(0);
        double rdot_z = rdot_vec(1);

        // compute the leg angular velocity
        double thetadot = (r_z * rdot_x - r_x * rdot_z) / (r_norm * r_norm);

        // leg length command
        double l0_command = x(4);
        
        // compute the ground reaction force
        Vector_2d lambd_vec = -r_hat * (k * (l0_command - r_norm) - b * (-v_com.dot(r_hat)));

        // compute the equivalent force from ankle torque (force prependicular to the leg applied at COM)
        if (this->params.torque_ankle == true) {
            
            // actual angle state and command theta states
            double theta = -std::atan2(r_x, -r_z);
            double theta_command = x(5);
            double thetadot_command = u(1);
            
            // compute the ankle torque
            double kp = this->params.torque_ankle_kp;
            double kd = this->params.torque_ankle_kd;
            double tau_ankle = kp * (theta_command - theta) + kd * (thetadot_command - thetadot);

            // saturate the ankle torque
            tau_ankle = std::max(-this->params.torque_ankle_lim, std::min(this->params.torque_ankle_lim, tau_ankle));

            // compute the equivalent force from ankle torque
            Vector_2d f_unit;
            double f_mag = tau_ankle / r_norm;
            f_unit << std::cos(theta), -std::sin(theta);
            f_com = f_mag * f_unit;
        }

        // no ankle torque applied
        else if (this->params.torque_ankle == false) {
            Vector_2d f_com;
            f_com << 0, 0;
        }

        // compute the stance dynamics
        a_com = (1/m) * lambd_vec + (1/m) * f_com + g_vec;

        // compute the leg dynamics
        v_leg << u(0), u(1);

        // stack the acceleration and velocity vectors
        xdot << v_com, a_com, v_leg;
    }

    else {
        std::cout << "Invalid domain for integration." << std::endl;
    }

    return xdot;
}


// compute the leg state
Vector_4d Dynamics::compute_leg_state(Vector_6d x_sys, Vector_2d p_foot, Vector_2d u, Domain d)
{
    // leg state variable
    Vector_4d x_leg;

    // indivudual states
    double r, theta, rdot, thetadot;

    // in flight (single integrator dynamics)
    if (d == Domain::FLIGHT) {
       
        // leg positions are the integrated velocity commands
        r = x_sys(4);
        theta = x_sys(5);
        rdot = u(0);
        thetadot = u(1);

        // populate the polar state vector
        x_leg << r, theta, rdot, thetadot;
    }

    // on ground (leg state goverened by the COM dynamics)
    else if (d == Domain::GROUND) {
       
        // unpack COM state
        Vector_2d p_com, v_com;
        p_com << x_sys(0), x_sys(1);
        v_com << x_sys(2), x_sys(3);

        // leg vectors
        Vector_2d r_vec, r_hat, rdot_vec;
        double r_norm, r_x, r_z, rdot_x, rdot_z;
        r_vec = p_foot - p_com;
        r_norm = r_vec.norm();
        r_hat = r_vec / r_norm;
        rdot_vec = -v_com;

        r_x = r_vec(0);
        r_z = r_vec(1);
        rdot_x = rdot_vec(0);
        rdot_z = rdot_vec(1);

        // compute the polar leg state 
        r = r_norm;
        rdot = -v_com.dot(r_hat);
        theta = -std::atan2(r_x, -r_z);
        thetadot = (r_z * rdot_x - r_x * rdot_z) / (r * r);

        // populate the polar state vector
        x_leg << r, theta, rdot, thetadot;
    }

    else {
        std::cout << "Invalid domain for computing leg state." << std::endl;
    }

    return x_leg;
}


// compute foot state in world frame
Vector_4d Dynamics::compute_foot_state(Vector_6d x_sys, Vector_4d x_leg, Vector_2d p_foot, Domain d)
{
    // foot state variable
    Vector_4d x_foot;

    // Flight (F), foot is in swing
    if (d == Domain::FLIGHT) {
        // com varaibles
        double px_com, pz_com, vx_com, vz_com;
        px_com = x_sys(0);
        pz_com = x_sys(1);
        vx_com = x_sys(2);
        vz_com = x_sys(3);

        // leg states
        double r, theta, rdot, thetadot;
        r = x_leg(0);
        theta = x_leg(1);
        rdot = x_leg(2);
        thetadot = x_leg(3);

        // compute the foot state via fwd kinematics
        double px_foot, pz_foot, vx_foot, vz_foot;
        px_foot = px_com - r * std::sin(theta);
        pz_foot = pz_com - r * std::cos(theta);
        vx_foot = vx_com - rdot * std::sin(theta) - r * thetadot * std::cos(theta);
        vz_foot = vz_com - rdot * std::cos(theta) + r * thetadot * std::sin(theta);

        // populate the foot state vector
        x_foot << px_foot, pz_foot, vx_foot, vz_foot;
    }

    // Ground (G), foot is in stance
    else if (d == Domain::GROUND) {
        // foot state is the same as the foot position w/ zero velocity
        x_foot << p_foot(0), p_foot(1), 0.0, 0.0;
    }

    else {
        std::cout << "Invalid domain for computing foot state." << std::endl;
    }

    return x_foot;
}


// Touch-Down (TD) Switching Surface -- cheks individual legs
bool Dynamics::S_TD(Vector_4d x_foot) {
   
    // unpack the state variables
    double pz_foot, vz_foot;
    pz_foot = x_foot(1);
    vz_foot = x_foot(3);

    // check the switching surface conditions
    bool gnd_pos, neg_vel, touchdown;
    gnd_pos = pz_foot <= 0.0;     // foot penetrated the ground
    neg_vel = vz_foot <= 0.0;     // foot is moving downward
    touchdown = gnd_pos && neg_vel; // if true, the foot touched down 

    return touchdown;
}

// Take-Off (TO) Switching Surface -- checks individual legs
bool Dynamics::S_TO(Vector_6d x_sys, Vector_4d x_leg) {
    
    // unpack the state variables
    double r, rdot, l0_command;
    r = x_leg(0);
    rdot = x_leg(2);
    l0_command = x_sys(4);

    // check the switching surface condition (zero force in leg)
    bool nom_length, pos_vel, takeoff;
    nom_length = r >= l0_command;  // leg is at or above the commanded length
    pos_vel = rdot >= 0.0;         // leg is moving upward
    takeoff = nom_length && pos_vel; // if true, the leg took off

    return takeoff;
}


// Check is a switching event occurred
Domain Dynamics::check_switching_event(Vector_6d x_sys, Vector_4d x_leg, Vector_4d x_foot, Domain d_current)
{
    // return the current domain after checking the switching surfaces
    Domain d_next;

    // In Flight (F)
    if (d_current == Domain::FLIGHT) {
        // check for a touchdown event
        bool touchdown = this->S_TD(x_foot);

        // if touchdown, switch to ground domain
        d_next = touchdown ? Domain::GROUND : Domain::FLIGHT;
    }

    // In Ground (G)
    else if (d_current == Domain::GROUND) {
        // check for a takeoff event
        bool takeoff = this->S_TO(x_sys, x_leg);

        // if takeoff, switch to flight domain
        d_next = takeoff ? Domain::FLIGHT : Domain::GROUND;
    }

    else {
        std::cout << "Invalid domain for checking switching event." << std::endl;
    }

    return d_next;
}


// apply the reset map
void Dynamics::reset_map(Vector_6d& x_sys, Vector_4d& x_leg, Vector_4d& x_foot, Vector_2d u, Domain d_prev, Domain d_post)
{
    // states to apply reset map to
    Vector_6d x_sys_post;
    Vector_4d x_leg_post;
    Vector_4d x_foot_post;
    Vector_2d p_foot_post;

    // Flight to Ground reset map
    if (d_prev == Domain::FLIGHT && d_post == Domain::GROUND) {
        
        // update the foot location (based on heuristic)
        p_foot_post << x_foot(0), 0.0;

        // update the leg state
        x_leg_post = this->compute_leg_state(x_sys, p_foot_post, u, d_post);

        // update the system state
        x_sys_post = x_sys;
        x_sys_post(4) = x_leg_post(0); // reset leg length command to the actual leg length at TD
        x_sys_post(5) = x_leg_post(1); // reset leg angle command to the actual leg angle at TD

        // update the foot state
        x_foot_post = this->compute_foot_state(x_sys_post, x_leg_post, p_foot_post, d_post);
    }

    // Ground to Flight reset amp
    else if (d_prev == Domain::GROUND && d_post == Domain::FLIGHT) {
        
        // update the foot location
        p_foot_post << x_foot(0), x_foot(1);  // NOTE: deviates from python implementation

        // update the leg state
        x_leg_post = x_leg;
        x_leg_post(2) = u(0); // leg velocity is the commanded velocity
        x_leg_post(3) = u(1); // leg angle velocity is the commanded velocity

        // update the system state
        x_sys_post = x_sys;
        x_sys_post(4) = x_leg_post(0); // reset leg length command to the actual leg length at TD
        x_sys_post(5) = x_leg_post(1); // reset leg angle command to the actual leg angle at TD

        // update the foot state
        x_foot_post = this->compute_foot_state(x_sys, x_leg, p_foot_post, d_post);
    }

    // update the states
    x_sys = x_sys_post;
    x_leg = x_leg_post;
    x_foot = x_foot_post;
}


// interpolate an input signal
Vector_2d Dynamics::interpolate_control_input(double t, Vector_1d_Traj T_u, Vector_2d_Traj U)
{
    // interpolated control input
    Vector_2d u;
    u.setZero();

    // find the first element in the time vector that is greater than the current time
    auto it = std::upper_bound(T_u.begin(), T_u.end(), t); 
    int idx = std::distance(T_u.begin(), it) - 1;  // return -1 if before the first element, 
                                                   // return N-1 if after the last element
    
    // Zero order hold (Z)
    if (this->params.interp == 'Z') {
        // constant control input
        u = U[idx];
    }

    // Linear interpolation (L)
    else if (this-> params.interp == 'L') {
        
        // beyond the last element
        if (idx == T_u.size() - 1) {
            u = U[idx];
        }
        // before the last element
        else if (idx == -1) {
            u = U[0];
        }
        // within an time interval
        else {
            // knot ends
            double t0 = T_u[idx];
            double tf = T_u[idx+1];
            Vector_2d u0 = U[idx];
            Vector_2d uf = U[idx+1];

            // linear interpolation
            u = u0 + (uf - u0) * (t - t0) / (tf - t0);
        }
    }
    else {
        std::cout << "Invalid interpolation method." << std::endl;
    }

    return u;
}


// RK3 integration function
Solution Dynamics::RK3_rollout(Vector_1d_Traj T_x, Vector_1d_Traj T_u, 
                               Vector_6d x0_sys, Vector_2d p0_foot, Domain d0,
                               Vector_2d_Traj U) 
{
    // integration parameters
    double dt = T_x[1] - T_x[0];
    int N = T_x.size();

    // make solution trajectory containers
    Vector_6d_Traj x_sys_t;   // system state trajectory
    Vector_4d_Traj x_leg_t;   // leg state trajectory
    Vector_4d_Traj x_foot_t;  // foot state trajectory
    Vector_2d_Traj u_t;       // interpolated control input trajectory
    Domain_Traj domain_t;     // domain trajectory
    x_sys_t.resize(N);
    x_foot_t.resize(N);
    x_leg_t.resize(N);
    u_t.resize(N);
    domain_t.resize(N);

    // initial conditions
    Vector_4d x0_leg = this->compute_leg_state(x0_sys, p0_foot, U[0], d0);
    Vector_4d x0_foot = this->compute_foot_state(x0_sys, x0_leg, p0_foot, d0);
    x_sys_t[0] = x0_sys;
    x_leg_t[0] = x0_leg;
    x_foot_t[0] = x0_foot;
    u_t[0] = U[0];
    domain_t[0] = d0;

    // current state variables
    Vector_6d xk_sys = x0_sys;
    Vector_4d xk_leg = x0_leg;
    Vector_4d xk_foot = x0_foot;
    Vector_2d p_foot = p0_foot;
    Vector_2d uk = U[0];
    Domain dk = d0;
    Domain dk_next;

    // ****************************** RK3 integration ******************************
    // viability variable (for viability kernel)
    bool viability = true;

    // intermediate times, inputs, and vector field values
    double tk, t1, t2, t3;
    Vector_2d u1, u2, u3;
    Vector_6d f1, f2, f3;

    // start RK3 integration
    for (int k = 1; k < N; k++) {
        // std::cout << "Integrating step: " << k << std::endl;

        // interpolation times
        tk = k * dt;
        t1 = tk;
        t2 = tk + 0.5 * dt;
        t3 = tk + dt;

        // interpolate the control input
        u1 = this->interpolate_control_input(t1, T_u, U);
        u2 = this->interpolate_control_input(t2, T_u, U);
        u3 = this->interpolate_control_input(t3, T_u, U);

        // vector fields for the RK3 integration
        f1 = this->dynamics(xk_sys, 
                            u1, p_foot, dk);
        f2 = this->dynamics(xk_sys + 0.5 * dt * f1, 
                            u2, p_foot, dk);
        f3 = this->dynamics(xk_sys - dt * f1 + 2 * dt * f2,
                            u3, p_foot, dk);

        // take the RK3 step
        xk_sys = xk_sys + (dt / 6) * (f1 + 4 * f2 + f3);
        xk_leg = this->compute_leg_state(xk_sys, p_foot, uk, dk);
        xk_foot = this->compute_foot_state(xk_sys, xk_leg, p_foot, dk);

        // check for switching events
        dk_next = this->check_switching_event(xk_sys, xk_leg, xk_foot, dk);

        // if there was a switching event, apply the reset map
        if (dk_next != dk) {

            // TODO: implement viability kernel

            // update all the states
            this->reset_map(xk_sys, xk_leg, xk_foot, u3, dk, dk_next);

            // update the foot location
            p_foot(0) = xk_foot(0);
            p_foot(1) = xk_foot(1);

            // update the domain
            dk = dk_next;
        }

        // store the states
        x_sys_t[k] = xk_sys;
        x_leg_t[k] = xk_leg;
        x_foot_t[k] = xk_foot;
        u_t[k] = u3;
        domain_t[k] = dk;
    }

    // pack the solution into the solution struct
    Solution sol;
    sol.t = T_x;
    sol.x_sys_t = x_sys_t;
    sol.x_leg_t = x_leg_t;
    sol.x_foot_t = x_foot_t;
    sol.u_t = u_t;
    sol.domain_t = domain_t;
    sol.viability = viability;

    return sol;
}
