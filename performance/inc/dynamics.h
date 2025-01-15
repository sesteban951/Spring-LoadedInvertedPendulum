// standard includes
#include <iostream>
#include "Eigen/Dense"

struct SystemParams
{
    double m;            // mass [kg]
    double g;            // gravity [m/s^2]
    double k;            // spring constant [N/m]
    double b;            // damping constant [Ns/m]
    double l0;           // nominal rest length [m]
    double r_min;        // minimum rest length [m]
    double r_max;        // maximum rest length [m]
    double theta_min;    // minimum leg angle from vertical [rad]
    double theta_max;    // maximum leg angle from vertical [rad]
    double rdot_lim;     // maximum leg extension velocity [m/s]
    double thetadot_lim; // maximum leg angle velocity [rad/s]
    bool torque_ankle;   // enable ankle torque 
    double torque_ankle_kp;           // proportional gain for ankle torque
    double torque_ankle_kd;           // derivative gain for ankle torque
};

// constructor
class Dynamics
{
    public:
        // Constructor
        Dynamics(SystemParams params_);  

        // System parameters
        SystemParams params;
};