// standard includes
#include <iostream>
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"
# include <vector>

// custom includes
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

    return 0;
}
