#include "../inc/dynamics.h"

Dynamics::Dynamics(YAML::Node config_file)
{
    // set the system parameters
    params.m = config_file["SYS_PARAMS"]["m"].as<double>();
    params.g = config_file["SYS_PARAMS"]["g"].as<double>();
    params.k = config_file["SYS_PARAMS"]["k"].as<double>();
    params.b = config_file["SYS_PARAMS"]["b"].as<double>();
    params.l0 = config_file["SYS_PARAMS"]["l0"].as<double>();
    params.r_min = config_file["SYS_PARAMS"]["r_min"].as<double>();
    params.r_max = config_file["SYS_PARAMS"]["r_max"].as<double>();
    params.theta_min = config_file["SYS_PARAMS"]["theta_min"].as<double>();
    params.theta_max = config_file["SYS_PARAMS"]["theta_max"].as<double>();
    params.rdot_lim = config_file["SYS_PARAMS"]["rdot_lim"].as<double>();
    params.thetadot_lim = config_file["SYS_PARAMS"]["thetadot_lim"].as<double>();
    params.torque_ankle = config_file["SYS_PARAMS"]["torque_ankle"].as<bool>();
    params.torque_ankle_lim = config_file["SYS_PARAMS"]["torque_ankle_lim"].as<double>();
    params.torque_ankle_kp = config_file["SYS_PARAMS"]["torque_ankle_kp"].as<double>();
    params.torque_ankle_kd = config_file["SYS_PARAMS"]["torque_ankle_kd"].as<double>();
}

State_Vec Dynamics::dynamics(State_Vec x, Control_Vec u, Foot_Pos_Vec p_foot, Domain d)
{
    // access some system parameters
    double m = this->params.m;
    double g = this->params.g;
    double k = this->params.k;
    double b = this->params.b;

    // unpack the state vector
    Eigen::Vector<double, 2> p_com;
    Eigen::Vector<double, 2> v_com;
    p_com << x(0), x(1);
    v_com << x(2), x(3);

    // vectors to use in the calculations
    Eigen::Vector<double, 2> a_com;
    Eigen::Vector<double, 2> v_leg;
    Eigen::Vector<double, 2> g_vec;
    g_vec << 0, -g;
    State_Vec xdot;

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
        Eigen::Vector<double, 2> r_vec, r_hat, rdot_vec, lambd;
        double r_norm, r_x, r_z, rdot_x, rdot_z, l0_command, tau_ankle;

        // compute the leg state
        r_vec = p_foot - p_com;
        r_norm = r_vec.norm();
        r_hat = r_vec / r_norm;
        r_x = r_vec(0);
        r_z = r_vec(1);

        rdot_vec = -v_com;
        rdot_x = rdot_vec(0);
        rdot_z = rdot_vec(1);

        // compute the leg angular velocity
        double thetadot = (r_z * rdot_x - r_x * rdot_z) / (r_norm * r_norm);

        // leg length command
        l0_command = u(0);
        
        // compute the groudn reaction force
        lambd = -r_hat * (k * (l0_command - r_norm) - b * (-v_com.dot(r_hat)));

        // compute the equivalent force from ankle torque (force prependicular to the leg applied at COM)
        Eigen::Vector<double, 2> f_com;
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
            Eigen::Vector<double, 2> f_com, f_unit;
            double f_mag = tau_ankle / r_norm;
            f_unit << std::cos(theta), -std::sin(theta);
            f_com = f_mag * f_unit;
        }

        // no ankle torque applied
        else if (this->params.torque_ankle == false) {
            f_com << 0, 0;
        }

        // compute the stance dynamics
        a_com = (1/m) * lambd + (1/m) * f_com + g_vec;

        // compute the leg dynamics
        v_leg << u(0), thetadot;

        // stack the acceleration and velocity vectors
        xdot << v_com, a_com, v_leg;
    }
    else {
        std::cout << "Invalid domain for integration." << std::endl;
    }

    return xdot;
}
