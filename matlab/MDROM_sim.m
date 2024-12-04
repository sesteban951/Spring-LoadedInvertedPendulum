%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multi-Domain Reduced Order Model (MDROM)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% SLIP params
params.m = 22;           % CoM mass (Achilles mass 22 kg)
params.g = 9.81;         % gravity
params.k = 1000;         % spring constant
params.b = 1;            % damping constant
alpha_max_deg = 60;      % max foot angle from verticle [deg]
params.alpha_max = alpha_max_deg * (pi/180);  % max foot angle [rad]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% simulation parameters
freq = 150;
dt = 1/freq;
tspan = 0:dt:3.0;

% initial conditions
p0_com = [0.0;  % px
          0.5]; % pz

% arbitrarily choose foot locations
p0_foot_L = [-0.1; 0.0]; % left foot
p0_foot_R = [0.1; 0.0];  % right foot

[r, alpha] = cartesian_to_polar_foot(p0_com, p0_foot_L);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COORDINATE CONVERSION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% caretesian to polar for feet
% NOTE: angles only defined in (-90, 90) deg angles.
function [r, alpha] = cartesian_to_polar_foot(p_com, p_foot)

    % compute the leg length
    r_rel = p_foot - p_com;
    r = norm(r_rel, 2);

    % compute the angle
    alpha = atan2(r_rel(2), r_rel(1)) + pi/2;
    alpha = -alpha;
end

% 


