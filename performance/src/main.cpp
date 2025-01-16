// standard includes
#include <iostream>
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"

// custom includes
#include "../inc/types.h"
#include "../inc/dynamics.h"
#include "../inc/control.h"

int main()
{
    // load parameters from yaml file
    YAML::Node config_file = YAML::LoadFile("../config/config.yaml");
    
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

    // print the results
    std::cout << "xleg: " << x_leg << std::endl;

    return 0;
}
