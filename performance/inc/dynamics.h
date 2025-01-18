#ifndef DYNAMICS_H
#define DYNAMICS_H

// standard includes
#include <iostream>

// package includes
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

        // NonLinear System dynamics, xdot = f(x, u, d)
        Vector_6d dynamics(Vector_6d x, 
                           Vector_2d u, 
                           Vector_2d p_foot,
                           Domain d);

        // compute the leg state local to COM frame
        Vector_4d compute_leg_state(Vector_6d x_sys, 
                                    Vector_2d p_foot, 
                                    Vector_2d u, 
                                    Domain d);

        // compute the foot state in world frame
        Vector_4d compute_foot_state(Vector_6d x_sys, 
                                     Vector_4d x_leg, 
                                     Vector_2d p_foot, 
                                     Domain d);

        // Switching Surfaces
        bool S_TD(Vector_4d x_foot);                 // Touch-Down (TD) Switching Surface
        bool S_TO(Vector_6d x_sys, Vector_4d x_leg); // Take-Off (TO) Switching Surface

        // event detection 
        Domain check_switching_event(Vector_6d x_sys, 
                                     Vector_4d x_leg, 
                                     Vector_4d x_foot, 
                                     Domain d);

        // reset map
        void reset_map(Vector_6d& x_sys, 
                       Vector_4d& x_leg, 
                       Vector_4d& x_foot, 
                       Vector_2d u, 
                       Domain d_prev, 
                       Domain d_post);

        // interpolate the control input
        Vector_2d interpolate_control_input(double t, 
                                            Vector_1d_Traj T_u, 
                                            Vector_2d_Traj U);
        
        // RK3 integration function, returns solutionof dynamics
        Solution RK3_rollout(Vector_1d_Traj T_x, Vector_1d_Traj T_u, 
                             Vector_6d x0, Vector_2d p0, Domain d0,
                             Vector_2d_Traj U);

    // private:
        // System parameters
        SystemParams params;
};

#endif