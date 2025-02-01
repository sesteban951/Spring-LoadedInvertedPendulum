%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ankle torque
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% generate random position
r_x_lims = [-0.5, 0.5];
r_y_lims = [-0.5, 0.5];
r_z_lims = [0.5, 1];
r = unifrnd([r_x_lims(1), r_y_lims(1), r_z_lims(1)], [r_x_lims(2), r_y_lims(2), r_z_lims(2)], 1, 3);
r = r';
r_x = r(1);
r_y = r(2);
r_z = r(3);

% generate random data
tau_x_lims = [-1, 1];
tau_y_lims = [-1, 1];
tau_xy = unifrnd([tau_x_lims(1), tau_y_lims(1)], [tau_x_lims(2), tau_y_lims(2)], 1, 2);
tau_xy = tau_xy';

% matrix
r_mag = norm(r);
sigma1 = r_mag^2 * r_z;
sigma2 = r_mag^2;
A_inv = [(r_x * r_y)/sigma1,      (r_y^2 + r_z^2)/sigma1, r_x/sigma2;
        -(r_x^2 + r_z^2)/sigma1, (-r_x * r_y)/sigma1,     r_y/sigma2;
          r_y/sigma2,             -r_x/sigma2,            r_z/sigma2];
F = A_inv * [tau_xy; 0];

% compute tau_z
tau_z = (r_x * F(2) - r_y * F(1));


% compute tau
tau = [tau_xy; tau_z];

% check conditions
disp("tau_xy =")
disp(tau_xy)
disp("||tau_xy|| =")
disp(norm(tau_xy))

disp("r = ")
disp(r)

disp("F = ")
disp(F)

disp("tau_z = ")
disp(tau_z)

disp("r x F = ")
disp(cross(r, F))

disp("r dot F = ")
disp(dot(r, F))

% plot a line
figure;
grid on; hold on;
axis equal;
azimuth = 45; % degrees
elevation = 30; % degrees
view(azimuth, elevation);
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z');

% plot the torque vector
quiver3(0, 0, 0, tau(1), tau(2), tau(3), 'c', 'LineWidth', 2);

% plot the position vector
plot3([0, r(1)], [0, r(2)], [0, r(3)], 'm', 'LineWidth', 2);
plot3([r(1)], [r(2)], [r(3)], 'ro', 'MarkerFaceColor', 'm', 'MarkerSize', 30);

% plot the force vector at the COM body
quiver3(r(1), r(2), r(3), F(1), F(2), F(3), 'k', 'LineWidth', 2);

% Define plane limits
x_lims = [-0.5, 0.5];
y_lims = [-0.5, 0.5];

% Create a mesh grid
[x_plane, y_plane] = meshgrid(linspace(x_lims(1), x_lims(2), 20), linspace(y_lims(1), y_lims(2), 20));

% Compute corresponding z values from the plane equation
z_plane = (r_x^2 + r_y^2 + r_z^2 - r_x * x_plane - r_y * y_plane) / r_z;

% Plot the plane
surf(x_plane, y_plane, z_plane, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'y');

% Reapply the axis labels and view
xlabel('X'); ylabel('Y'); zlabel('Z');

set(gcf,'renderer','painters')

