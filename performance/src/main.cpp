// standard includes
#include <iostream>
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"

// custom includes
#include "../inc/dynamics.h"

int main()
{
    // load parameters from yaml file
    YAML::Node sys_config = YAML::LoadFile("../config/config.yaml");
    SystemParams sys_params;
    sys_params.m = sys_config["SYS_PARAMS"]["m"].as<double>();
    sys_params.g = sys_config["SYS_PARAMS"]["g"].as<double>();
    sys_params.k = sys_config["SYS_PARAMS"]["k"].as<double>();
    sys_params.b = sys_config["SYS_PARAMS"]["b"].as<double>();
    sys_params.l0 = sys_config["SYS_PARAMS"]["l0"].as<double>();
    sys_params.r_min = sys_config["SYS_PARAMS"]["r_min"].as<double>();
    sys_params.r_max = sys_config["SYS_PARAMS"]["r_max"].as<double>();
    sys_params.theta_min = sys_config["SYS_PARAMS"]["theta_min"].as<double>();
    sys_params.theta_max = sys_config["SYS_PARAMS"]["theta_max"].as<double>();
    sys_params.rdot_lim = sys_config["SYS_PARAMS"]["rdot_lim"].as<double>();
    sys_params.thetadot_lim = sys_config["SYS_PARAMS"]["thetadot_lim"].as<double>();
    sys_params.torque_ankle = sys_config["SYS_PARAMS"]["torque_ankle"].as<bool>();
    sys_params.torque_ankle_kp = sys_config["SYS_PARAMS"]["torque_ankle_kp"].as<double>();
    sys_params.torque_ankle_kd = sys_config["SYS_PARAMS"]["torque_ankle_kd"].as<double>();

    // create dynamics object
    Dynamics dynamics(sys_params);

    double a = dynamics.params.g;
    std::cout << a << std::endl;

    return 0;
}