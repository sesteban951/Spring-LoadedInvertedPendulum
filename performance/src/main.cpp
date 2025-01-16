// standard includes
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>

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
    Vector_6d x;
    Vector_2d u;
    Vector_2d p_foot;
    Domain d;

    // initial conditions
    x << 0,    // px_com
         0.75, // pz_com
         1,    // vx_com
         1,    // vz_com
         0,    // l0_command
         0;    // theta_command
    u << 0.1,    // l0dot_command
         0.2;    // thetadot_command
    p_foot << 0.1,  // px_foot
              0.1;  // pz_foot
    d = Domain::FLIGHT;

    // query the dynamics
    Vector_6d xdot = dynamics.dynamics(x, u, p_foot, d);
    
    // query the leg state
    Vector_4d x_leg = dynamics.compute_leg_state(x, p_foot, u, d);

    // query the foot state
    Vector_4d x_foot = dynamics.compute_foot_state(x, x_leg, p_foot, d);

    // print the results
    std::cout << "xdot: " << xdot << std::endl;
    std::cout << "xleg: " << x_leg << std::endl;
    std::cout << "xfoot: " << x_foot << std::endl;

    // build a time vector
    Vector_1d_Traj T_x, T_u;
    Vector_2d_Traj U;
    T_x.resize(controller.params.N);
    T_u.resize(controller.params.Nu);
    U.resize(controller.params.Nu);  

    std::cout << "T_x: " << T_x.size() << std::endl;
    std::cout << "T_u: " << T_u.size() << std::endl;
    std::cout << "U: " << U.size() << std::endl;

    // print the trajectories
    Solution sol;
    auto t0 = std::chrono::high_resolution_clock::now();
    sol = dynamics.RK3_rollout(T_x, T_u, x, p_foot, d, U);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Time to integrate: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " microseconds" << std::endl;

    // where to save each trajectory
    std::string x_sys_file = "../data/state_sys.csv";
    std::string x_leg_file = "../data/state_leg.csv";
    std::string x_foot_file = "../data/state_foot.csv";
    std::string u_file = "../data/input.csv";
    std::string domain_file = "../data/domain.csv";

    // save the solution to a file
    std::ofstream file;
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
