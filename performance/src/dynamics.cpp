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
    params.torque_ankle_kp = config_file["SYS_PARAMS"]["torque_ankle_kp"].as<double>();
    params.torque_ankle_kd = config_file["SYS_PARAMS"]["torque_ankle_kd"].as<double>();
}

Dynamics::dynamics(state_vec x, control_vec u, domain d)
{
    // access some system parameters
    double m = params.m;
    double g = params.g;
    double k = params.k;
    double b = params.b;

    // unpack the state vector
    Eigen::Vector<double, 2> p_com;
    Eigen::Vector<double, 2> v_com;
}
