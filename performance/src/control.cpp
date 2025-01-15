#include "../inc/control.h"

Controller::Controller(YAML::Node config_file)
{
    // set the control parameters
    params.N = config_file["CTRL_PARAMS"]["N"].as<int>();
    params.dt = config_file["CTRL_PARAMS"]["dt"].as<double>();
    params.K = config_file["CTRL_PARAMS"]["K"].as<int>();
    params.Nu = config_file["CTRL_PARAMS"]["Nu"].as<int>();
    params.interp = config_file["CTRL_PARAMS"]["interp"].as<char>();
    std::vector<double> Q_diags_temp = config_file["CTRL_PARAMS"]["Q_diags"].as<std::vector<double>>();
    std::vector<double> Qf_diags_temp = config_file["CTRL_PARAMS"]["Qf_diags"].as<std::vector<double>>();
    std::vector<double> R_diags_temp = config_file["CTRL_PARAMS"]["R_diags"].as<std::vector<double>>();
    params.Q_diags = Eigen::Map<Eigen::VectorXd>(Q_diags_temp.data(), Q_diags_temp.size());
    params.Qf_diags = Eigen::Map<Eigen::VectorXd>(Qf_diags_temp.data(), Qf_diags_temp.size());
    params.R_diags = Eigen::Map<Eigen::VectorXd>(R_diags_temp.data(), R_diags_temp.size());
    params.N_elite = config_file["CTRL_PARAMS"]["N_elite"].as<int>();
    params.CEM_iters = config_file["CTRL_PARAMS"]["CEM_iters"].as<int>();
}

