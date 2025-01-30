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
% r(2) = 0;
r_x = r(1);
r_y = r(2);
r_z = r(3);

% compute euler angles
euler_angles = vectorToEuler_RollPitch(r)

% generate random data
tau_x_lims = [-1, 1];
tau_y_lims = [-1, 1];
tau_xy = unifrnd([tau_x_lims(1), tau_y_lims(1)], [tau_x_lims(2), tau_y_lims(2)], 1, 2);

%  build the R^3 vector
tau = [tau_xy(1), tau_xy(2), 0]';
% tau = [0, tau_xy(2), 0]';
tau_x = tau(1);
tau_y = tau(2);
tau_z = tau(3);

% Chat gpt
r_cross = [  0, -r(3),  r(2);
            r(3),  0, -r(1);
           -r(2), r(1),  0];
% F = pinv(r_cross) * tau;

% equivalent force applied to COM body
F_z = (tau_x * r_y - tau_y * r_x) / (r_x^2 + r_y^2 + r_z^2);
F_x = (r_x/r_z) * F_z + tau_y/r_z;
F_y = (r_y/r_z) * F_z - tau_x/r_z;
F = [F_x, F_y, F_z]';

% alterantive
% r_unit = r/norm(r);
% tau_unit = tau/norm(tau);
% f_unit = cross(tau_unit, r_unit);
% F = f_unit * norm(tau);

% alterantive 2
% A = [r_cross; r'];
% A_left_inv  = inv(A' * A) * A';
% F = A_left_inv * [tau; 0];

% check conditions
disp("tau =")
disp(tau)
disp("||tau|| =")
disp(norm(tau))

disp("r: ")
disp(r)

disp("r x F: ")
% r_z_vec = [0,0,r_z];
% disp(cross(r_z_vec, F))
disp(cross(r, F))

disp("r dot F = 0?")
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


function euler_angles = vectorToEuler_RollPitch(v)
    % Ensure the vector is normalized
    v = v / norm(v);
    
    % Extract components
    x = v(1);
    y = v(2);
    z = v(3);

    % Compute pitch (θ) from the Y-axis rotation
    pitch = atan2(-z, x);  % Rotation about Y-axis

    % Compute roll (φ) from the X-axis rotation
    roll = atan2(y, sqrt(x^2 + z^2));  % Rotation about X-axis

    % Return Euler angles [roll; pitch] in radians
    euler_angles = [roll; pitch];
end