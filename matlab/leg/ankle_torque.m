clear all; close all; clc;

% me trying to invert a torque to get required force
r_bounds = [-1, 1;     % [x_min, x_max]
            -1, -0.1]; % [z_min, z_max]

% generate one r vector
r = unifrnd(r_bounds(:,1), r_bounds(:,2), 2, 1)
r_norm = norm(r, 2);

% find the angle from the negative y-axis and clockwise
theta = -atan2(r(1), -r(2));
theta_deg = -atan2(r(1), -r(2)) * 180/pi

% generate one torque vector
tau_bounds = [-1, 1];
tau = unifrnd(tau_bounds(1), tau_bounds(2), 1, 1)
% tau_norm = norm(tau, 2)

% compute the force vector
f_unit = [cos(theta), -sin(theta)]';
f_mag = tau / r_norm;
f = f_mag * f_unit;
f_norm = norm(f, 2);
tau_check = r_norm * f_norm

% draw the r vector
figure;
hold on; 
xline(0);
yline(0);

% draw the f vector
plot([0, r(1)], [0, r(2)], 'r', 'LineWidth', 2);
plot(r(1), r(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% draw the f vector
quiver(0, 0, f(1), f(2), 'LineWidth', 2, 'MaxHeadSize', 0.1);

grid on; axis equal;
