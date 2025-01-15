#pragma once

#include<Eigen/Dense>
#include<vector>

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

// ***********************************************************************************
// ENUMS
// ***********************************************************************************

enum class Domain {FLIGHT, GROUND};

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
};

// struct to hold control parameters
struct ControlParams
{
    int N;               // number of system dynamics integration steps
    double dt;           // time step [sec]
    int K;               // number of parallel 
    int Nu;              // number of control points
    char interp;         // interpolation method
    Vector_8d Q_diags;   // diagonal elements of Q matrix
    Vector_8d Qf_diags;  // diagonal elements of Qf matrix
    Vector_2d R_diags;   // diagonal elements of R matrix
    int N_elite;         // number of elite control sequences
    int CEM_iters;       // number of CEM iterations
};

