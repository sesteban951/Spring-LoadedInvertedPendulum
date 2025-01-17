// standard includes
#include <iostream>
#include <random>

// package includes
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

        // to initialize the initial distribution
        void initialize_distribution(YAML::Node config_file);

        // sample a bundle of control inputs from the distribution
        void sample_input_trajectory(int K);

    // private:
        // Control parameters
        ControlParams params;
        GaussianDistribution dist;
};
