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
    Vector_6d x;
    Vector_2d u;
    Vector_2d p_foot;
    Domain d;

    // initial conditions
    x << 0.0,    // px_com
         0.75, // pz_com
         0.5,    // vx_com
         0.5,    // vz_com
         dynamics.params.l0,    // l0_command
         0.0;                   // theta_command
    p_foot << 0.0,  // px_foot
              0.0;  // pz_foot
    d = Domain::FLIGHT;

    // build a time vector
    Vector_1d_Traj T_x;
    T_x.resize(controller.params.N);
    for (int i = 0; i < controller.params.N; i++) {
        T_x[i] = i * controller.params.dt;
    }

    // build input trajectory
    Vector_1d_Traj T_u;
    Vector_2d_Traj U;
    Vector_2d U_const;
    T_u.resize(controller.params.Nu);
    U.resize(controller.params.Nu);
    U_const << 0.0, -0.5;
    for (int i = 0; i < controller.params.Nu; i++) {
        T_u[i] = i * controller.params.dt;
        U[i] = U_const;
    }


    // single rollout
    Solution sol;
    // auto t0 = std::chrono::high_resolution_clock::now();
    // sol = dynamics.RK3_rollout(T_x, T_u, x, p_foot, d, U);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::cout << "Time to integrate: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " microseconds" << std::endl;


    // generate a reference trajectory
    // Vector_8d_Traj X_ref;
    // X_ref = controller.generate_reference_trajectory(x.head<4>());

    // // evaluate the cost function
    // double J = controller.cost_function(X_ref, sol, U);

    // std::cout << "Cost: " << J << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    MC_Tuple mc_tuple;
    mc_tuple = controller.monte_carlo(x, p_foot, d);
    auto tf = std::chrono::high_resolution_clock::now();
    std::cout << "Time to integrate: " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count() << " microseconds" << std::endl;


    // // where to save each trajectory
    // std::string time_file = "../data/time.csv";
    // std::string x_sys_file = "../data/state_sys.csv";
    // std::string x_leg_file = "../data/state_leg.csv";
    // std::string x_foot_file = "../data/state_foot.csv";
    // std::string u_file = "../data/input.csv";
    // std::string domain_file = "../data/domain.csv";

    // // save the solution to a file
    // std::ofstream file;

    // file.open(time_file);
    // for (int i = 0; i < T_x.size(); i++) {
    //     file << T_x[i] << std::endl;
    // }
    // file.close();

    // file.open(x_sys_file);
    // for (int i = 0; i < sol.x_sys_t.size(); i++) {
    //     file << sol.x_sys_t[i].transpose() << std::endl;
    // }
    // file.close();

    // file.open(x_leg_file);
    // for (int i = 0; i < sol.x_leg_t.size(); i++) {
    //     file << sol.x_leg_t[i].transpose() << std::endl;
    // }
    // file.close();

    // file.open(x_foot_file);
    // for (int i = 0; i < sol.x_foot_t.size(); i++) {
    //     file << sol.x_foot_t[i].transpose() << std::endl;
    // }
    // file.close();

    // file.open(u_file);
    // for (int i = 0; i < sol.u_t.size(); i++) {
    //     file << sol.u_t[i].transpose() << std::endl;
    // }
    // file.close();

    // file.open(domain_file);
    // for (int i = 0; i < sol.domain_t.size(); i++) {
    //     if (sol.domain_t[i] == Domain::FLIGHT) {
    //         file << 0 << std::endl;
    //     }
    //     else if (sol.domain_t[i] == Domain::GROUND) {
    //         file << 1 << std::endl;
    //     }
    // }
    // file.close();

    return 0;
}
