%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testing rotations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% extrinsic rotation angles
theta_x_lims = [-80, 80];
theta_y_lims = [-80, 80];
theta_x_deg = unifrnd(theta_x_lims(1), theta_x_lims(2));
theta_y_deg = unifrnd(theta_y_lims(1), theta_y_lims(2));
% theta_x_deg = 45;
% theta_y_deg = 45;
theta_x = deg2rad(theta_x_deg);
theta_y = deg2rad(theta_y_deg);

% compute the absolute angle
L = 1;
r0 = [0, 0, -L];
rz = -sqrt(L^2 / (tan(theta_x)^2 + tan(theta_y)^2 + 1));
rx = rz * tan(theta_y);
ry = -rz * tan(theta_x);
r_rot = [rx, ry, rz];

% projections on the axes
r_theta_yz = [0, sin(theta_x), -cos(theta_x)];
r_theta_xz = [-sin(theta_y), 0, -cos(theta_y)];


% display the results
disp("random theta_x =")
disp(theta_x)
disp("random theta_y =")
disp(theta_y)

theta_x_computed = atan2(ry, -rz);
theta_y_computed = -atan2(rx, -rz);

disp("computed theta_x =")
disp(theta_x_computed)
disp("computed theta_y =")
disp(theta_y_computed)

% plot it in 3D
figure;
grid on; hold on; 
axis equal;

%  plot axis
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);

% plot initial vector
plot3([0, r0(1)], [0, r0(2)], [0, r0(3)], 'k--', 'LineWidth', 2);

% plot rotated vector
plot3([0, r_rot(1)], [0, r_rot(2)], [0, r_rot(3)], 'm', 'LineWidth', 2);

% plot projections
plot3([0, r_theta_yz(1)], [0, r_theta_yz(2)], [0, r_theta_yz(3)], 'k:', 'LineWidth', 2);
plot3([0, r_theta_xz(1)], [0, r_theta_xz(2)], [0, r_theta_xz(3)], 'k:', 'LineWidth', 2);

msg = sprintf('theta_x = %.3f\ntheta_y = %.3f', theta_x_deg, theta_y_deg);
title(msg);

xlabel('X');
ylabel('Y');
zlabel('Z');
view(3);
