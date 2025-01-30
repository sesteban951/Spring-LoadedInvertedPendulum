%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3D KINEMATICS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

angle_x = 45; % degrees
angle_y = 30; % degrees

% convert to radians
angle_x = angle_x * pi/180; % degrees to radians
angle_y = angle_y * pi/180; % degrees to radians

% original vector
r0 = [0, 0, -1]';

% rotation matrix
R = R_y(angle_y) * R_x(angle_x);

% rotated vector
r = R * r0;
r_x = r(1);
r_y = r(2);
r_z = r(3);

% angle from negative z-axis
pitch = -atan2(r_x, -r_z) * 180/pi;
roll = atan2(r_y, -r_z) * 180/pi;

% plot a line
figure;

% plot the x, y, z axes
subplot(2, 2, 2);
grid on; hold on;
axis equal;
azimuth = 45; % degrees
elevation = 30; % degrees
view(azimuth, elevation);
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z');
plot3([0, r0(1)], [0, r0(2)], [0, r0(3)], 'k--', 'LineWidth', 2);
plot3([0, r(1)], [0, r(2)], [0, r(3)], 'k', 'LineWidth', 2);

% plot the original vector
subplot(2, 2, 1);
grid on; hold on;
axis equal;
azimuth = 0;    % degrees
elevation = 90; % degrees
view(azimuth, elevation);
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z');
plot3([0, r0(1)], [0, r0(2)], [0, r0(3)], 'k--', 'LineWidth', 2);
plot3([0, r(1)], [0, r(2)], [0, r(3)], 'k', 'LineWidth', 2);
% angle = atan2(-r_x, r_y) * 180/pi;
% title(strcat('Angle = ', num2str(angle, '%0.2f'), ' deg'));

subplot(2, 2, 3);
grid on; hold on;
axis equal;
azimuth = 0;   % degrees
elevation = 0; % degrees
view(azimuth, elevation);
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z');
plot3([0, r0(1)], [0, r0(2)], [0, r0(3)], 'k--', 'LineWidth', 2);
plot3([0, r(1)], [0, r(2)], [0, r(3)], 'k', 'LineWidth', 2);
title(strcat('Pitch = ', num2str(pitch, '%0.2f'), ' deg'));

subplot(2, 2, 4);
grid on; hold on;
axis equal;
azimuth = 90;   % degrees
elevation = 0; % degrees
view(azimuth, elevation);
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z');
plot3([0, r0(1)], [0, r0(2)], [0, r0(3)], 'k--', 'LineWidth', 2);
plot3([0, r(1)], [0, r(2)], [0, r(3)], 'k', 'LineWidth', 2);
title(strcat('Roll = ', num2str(roll, '%0.2f'), ' deg'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUXILIARY FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% rotation about x-axis
function R_x = R_x(angle)
    R_x = [1, 0,           0; 
           0, cos(angle), -sin(angle); 
           0, sin(angle),  cos(angle)];
end

% rotation about y-axis
function R_y = R_y(angle)
    R_y = [cos(angle),  0, sin(angle); 
           0,           1, 0; 
          -sin(angle),  0, cos(angle)];
end