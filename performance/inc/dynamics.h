// standard includes
#include <iostream>
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"
#include <vector>

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

// common typedefs
typedef Eigen::Vector<double, 6> State_Vec;
typedef Eigen::Vector<double, 2> Control_Vec;
typedef Eigen::Vector<double, 2> Foot_Pos_Vec;
typedef Eigen::Vector<double, 4> Foot_State_Vec;
enum class Domain {FLIGHT, GROUND};

// class for system dynamics
class Dynamics
{
    public:
        // Constructor
        Dynamics(YAML::Node config_file);  
        ~Dynamics(){};

        // System dynamics
        State_Vec dynamics(State_Vec x, 
                           Control_Vec u, 
                           Foot_Pos_Vec p_foot,
                           Domain d);

    private:
        // System parameters
        SystemParams params;
};