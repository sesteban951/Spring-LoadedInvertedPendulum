#ifndef TYPES_H
#define TYPES_H

// standard libraries
#include <Eigen/Dense>
#include <vector>
#include <tuple>

// ***********************************************************************************
// ENUMS
// ***********************************************************************************

enum class Domain {FLIGHT, GROUND};

// ***********************************************************************************
// ARRAYS
// ***********************************************************************************

// Dynamic arrays
using Vector_d = Eigen::Vector<double, Eigen::Dynamic>;
using Matrix_d = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

// Fixed size arrays
using Vector_2d = Eigen::Vector<double, 2>;
using Vector_4d = Eigen::Vector<double, 4>;
using Vector_6d = Eigen::Vector<double, 6>;
using Vector_8d = Eigen::Vector<double, 8>;

using Matrix_2d = Eigen::Matrix<double, 2, 2>;
using Matrix_8d = Eigen::Matrix<double, 8, 8>;

// Time Series Types
using Vector_1i_Traj = std::vector<int>;
using Vector_1d_Traj = std::vector<double>;
using Vector_2d_Traj = std::vector<Vector_2d>;
using Vector_4d_Traj = std::vector<Vector_4d>;
using Vector_6d_Traj = std::vector<Vector_6d>;
using Vector_8d_Traj = std::vector<Vector_8d>;

using Vector_d_Traj = std::vector<Vector_d>;
using Matrix_d_Traj = std::vector<Matrix_d>;

using Domain_Traj = std::vector<Domain>;

// ***********************************************************************************
// STRUCTS
// ***********************************************************************************

// struct to store system parameters
struct SystemParams
{
    double m;                 // mass [kg]
    double g;                 // gravity [m/s^2]
    double k;                 // spring constant [N/m]
    double b;                 // damping constant [Ns/m]
    double l0;                // nominal rest length [m]
    double r_min;             // minimum rest length [m]
    double r_max;             // maximum rest length [m]
    double theta_min;         // minimum leg angle from vertical [rad]
    double theta_max;         // maximum leg angle from vertical [rad]
    double rdot_lim;          // maximum leg extension velocity [m/s]
    double thetadot_lim;      // maximum leg angle velocity [rad/s]
    bool torque_ankle;        // enable ankle torque 
    double torque_ankle_lim;  // enable ankle torque 
    double torque_ankle_kp;   // proportional gain for ankle torque
    double torque_ankle_kd;   // derivative gain for ankle torque
    char interp;              // control interpolation type
};

// struct to hold control parameters
struct ControlParams
{
    int N;           // number of system dynamics integration steps
    double dt;       // time step [sec]
    int K;           // number of parallel 
    int Nu;          // number of control points
    int N_elite;     // number of elite control sequences
    int CEM_iters;   // number of CEM iterations
    Matrix_8d Q;     // diagonal elements of Q matrix
    Matrix_8d Qf;    // diagonal elements of Qf matrix
    Matrix_2d R;     // diagonal elements of R matrix
};

// Distribution struct
struct GaussianDistribution
{
    Vector_d mean;      // mean of the distribution
    Matrix_d cov;       // covariance of the distribution
    double epsilon;     // small value to add to the diagonal of the covariance
    bool diag_cov;      // choose to enfoce diagonal covariance
    bool seed_enabled;  // enable random number generator seed
    int seed;           // random number generator seed
};

// dynamics solution struct
struct Solution
{
    Vector_1d_Traj t;        // time trajectory
    Vector_6d_Traj x_sys_t;  // system state trajectory
    Vector_4d_Traj x_leg_t;  // leg state trajectory
    Vector_4d_Traj x_foot_t; // foot state trajectory
    Vector_2d_Traj u_t;      // interpolated control input trajectory
    Domain_Traj domain_t;    // domain trajectory
    bool viability;          // viability of the trajectory
};

// ***********************************************************************************
// Bundles
// ***********************************************************************************

// Bundle of Trajectories
using Vector_2d_Traj_Bundle = std::vector<Vector_2d_Traj>;
using Vector_8d_Traj_Bundle = std::vector<Vector_8d_Traj>;
using Vector_d_Traj_Bundle = std::vector<Vector_d_Traj>;

// Bundle of Solutions
using Solution_Bundle = std::vector<Solution>;

// ***********************************************************************************
// Monte Carlo Result
// ***********************************************************************************

struct MC_Result
{
    Solution_Bundle S;        // Solutions
    Vector_2d_Traj_Bundle U;  // Control Inputs  
    Vector_1d_Traj J;         // Costs
};

#endif