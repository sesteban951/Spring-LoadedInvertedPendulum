// standard includes
#include <iostream>
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"

// custom includes
#include "types.h"

// class for my controller
class Controller
{
    public:
        // Constructor and Destructor
        Controller(YAML::Node config_file);  
        ~Controller(){};

    // private:
        // Control parameters
        ControlParams params;
};
