#ifndef CONTROL_H
#define CONTROL_H

// standard includes
#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

// package includes
#include "yaml-cpp/yaml.h"

// custom includes
#include "types.h"
#include "dynamics.h"

// class for my controller
class Controller
{
    public:
        // Constructor and Destructor
        Controller(YAML::Node config_file);
        ~Controller(){};

        // to initialize the initial distribution
        void initialize_distribution(YAML::Node config_file);

        // compute mean and covariance from a bundle of control inputs
        void update_dsitribution_params(Vector_2d_Traj_Bundle U_bundle);

        // sample a bundle of control inputs from the distribution
        Vector_2d_Traj_Bundle sample_input_trajectory(int K);

        // generate a reference trajectory for the predictive control to track
        Vector_8d_Traj generate_reference_trajectory(Vector_4d x0_com);

        // evaulate the cost function given a solution
        double cost_function(Vector_8d_Traj X_des, Solution Sol, Vector_2d_Traj U);

        // perform open loop rollouts
        MC_Result monte_carlo(Vector_6d x0_sys, Vector_2d p0_foot, Domain d0);

        // sort solutions based on cost
        void sort_trajectories(Solution_Bundle  S,       Vector_2d_Traj_Bundle  U,
                               Solution_Bundle& S_elite, Vector_2d_Traj_Bundle& U_elite,
                               Vector_1d_Traj J);

        // sampling predictive control scheme here
        Solution sampling_predictive_control(Vector_6d x0_sys, Vector_2d p0_foot, Domain d0);

    // private:
        // internal dynamics object
        Dynamics dynamics;

        // Control parameters
        ControlParams params;

        // distribution parameters
        GaussianDistribution dist;

        // random number generator
        std::mt19937 rand_generator;
        std::normal_distribution<double> normal_dist;

        // desired reference trajectory
        double pz_des;
        double vx_des;
        double r_des;
};

#endif