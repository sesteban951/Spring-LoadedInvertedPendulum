// standard includes
#include <iostream>
#include <chrono>
#include <fstream>

// package includes
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"

// custom includes
#include "../inc/types.h"
#include "../inc/dynamics.h"
#include "../inc/control.h"

int main()
{
    // load parameters from yaml file
    std::string config_file_path = "../config/config.yaml";   
    YAML::Node config_file = YAML::LoadFile(config_file_path);
    
    // create dynamics object
    Dynamics dynamics(config_file);

    // create the controller object
    Controller controller(config_file);

    // testing querying the dynamics
    Vector_6d x0;
    Vector_2d p0_foot;
    Domain d0;

    // initial conditions
    std::vector<double> x0_temp = config_file["STATE"]["x0"].as<std::vector<double>>();
    x0 << x0_temp[0],    // px_com
          x0_temp[1],    // pz_com
          x0_temp[2],    // vx_com
          x0_temp[3],    // vz_com
          x0_temp[4],    // l0_command
          x0_temp[5];    // theta_command
    p0_foot << 0.0,  // px_foot
               0.0;  // pz_foot
    d0 = Domain::FLIGHT;

    // perform monte carlo simulation
    Solution sol;
    auto t0 = std::chrono::high_resolution_clock::now();
    sol = controller.sampling_predictive_control(x0, p0_foot, d0);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    // unpack the solution
    Vector_1d_Traj T_x = sol.t;
    Vector_6d_Traj x_sys = sol.x_sys_t;
    Vector_4d_Traj x_leg = sol.x_leg_t;
    Vector_4d_Traj x_foot = sol.x_foot_t;
    Vector_2d_Traj u = sol.u_t;
    Domain_Traj domain = sol.domain_t;

    // where to save each trajectory
    std::string time_file = "../data/time.csv";
    std::string x_sys_file = "../data/state_sys.csv";
    std::string x_leg_file = "../data/state_leg.csv";
    std::string x_foot_file = "../data/state_foot.csv";
    std::string u_file = "../data/input.csv";
    std::string domain_file = "../data/domain.csv";

    // save the solution to a file
    std::ofstream file;

    file.open(time_file);
    for (int i = 0; i < T_x.size(); i++) {
        file << T_x[i] << std::endl;
    }
    file.close();

    file.open(x_sys_file);
    for (int i = 0; i < sol.x_sys_t.size(); i++) {
        file << sol.x_sys_t[i].transpose() << std::endl;
    }
    file.close();

    file.open(x_leg_file);
    for (int i = 0; i < sol.x_leg_t.size(); i++) {
        file << sol.x_leg_t[i].transpose() << std::endl;
    }
    file.close();

    file.open(x_foot_file);
    for (int i = 0; i < sol.x_foot_t.size(); i++) {
        file << sol.x_foot_t[i].transpose() << std::endl;
    }
    file.close();

    file.open(u_file);
    for (int i = 0; i < sol.u_t.size(); i++) {
        file << sol.u_t[i].transpose() << std::endl;
    }
    file.close();

    file.open(domain_file);
    for (int i = 0; i < sol.domain_t.size(); i++) {
        if (sol.domain_t[i] == Domain::FLIGHT) {
            file << 0 << std::endl;
        }
        else if (sol.domain_t[i] == Domain::GROUND) {
            file << 1 << std::endl;
        }
    }
    file.close();

    return 0;
}
