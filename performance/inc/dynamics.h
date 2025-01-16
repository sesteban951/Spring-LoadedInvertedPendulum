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

        // RK3 integration function
        void RK3_rollout(Vector_1d_Traj T_x, Vector_1d_Traj T_u, 
                         Vector_6d x0, Vector_2d p0, Domain d0,
                         Vector_2d_Traj U);

        // compute the leg state
        Vector_4d compute_leg_state(Vector_6d x, 
                                    Vector_2d p_foot, 
                                    Vector_2d u, 
                                    Domain d);

    // private:
        // System parameters
        SystemParams params;
};