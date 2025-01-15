// standard includes
#include <iostream>
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"

// custom includes
#include "types.h"

// class for system dynamics
class Dynamics
{
    public:
        // Constructor and Destructor
        Dynamics(YAML::Node config_file);  
        ~Dynamics(){};

        // System dynamics
        Vector_6d dynamics(Vector_6d x, 
                           Vector_2d u, 
                           Vector_2d p_foot,
                           Domain d);

    // private:
        // System parameters
        SystemParams params;
};