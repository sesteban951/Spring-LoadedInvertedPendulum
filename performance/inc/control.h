// standard includes
#include <iostream>
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"
#include <vector>

// struct to hold control parameters
struct ControlParams
{
    int N;                              // number of system dynamics integration steps
    double dt;                          // time step [sec]
    int K;                              // number of parallel 
    int Nu;                             // number of control points
    char interp;                        // interpolation method
    Eigen::Vector<double, 8> Q_diags;   // diagonal elements of Q matrix
    Eigen::Vector<double, 8> Qf_diags;  // diagonal elements of Qf matrix
    Eigen::Vector<double, 2> R_diags;   // diagonal elements of R matrix
    int N_elite;                        // number of elite control sequences
    int CEM_iters;                      // number of CEM iterations
};

// class for my controller
class Controller
{
    public:
        // Constructor amd Destructor
        Controller(YAML::Node config_file);  
        ~Controller(){};

    private:
        // Control parameters
        ControlParams params;
};
