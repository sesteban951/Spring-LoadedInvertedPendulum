%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testing rotations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% extrinsic rotation angles
theta_x = 1.0;
theta_y = 1.0;

% extrinsic rotation angles
L = 1;
r0 = [0, 0, -L];
% R_pitch_ = R_pitch(y_angle);
% R_roll_ = R_roll(x_angle);
% r_rot = R_roll_ * R_pitch_ * r0';
r_z = -sqrt(L^2 / (tan(theta_x)^2 + tan(theta_y)^2 + 1));
r_x = r_z * tan(theta_y);
r_y = -r_z * tan(theta_x);
r_rot = [r_x, r_y, r_z]

% find the absolute angles 
% theta_y = 
% theta_x = 
% theta_x = rad2deg(theta_x);
% theta_y = rad2deg(theta_y);

% x_angle = rad2deg(x_angle);
% y_angle = rad2deg(y_angle);

% disp(['theta_x_extrinsic: ', num2str(x_angle)]);
% disp(['theta_y_extrinsic: ', num2str(x_angle)]);
% disp(['theta_x: ', num2str(theta_x)]);
% disp(['theta_y: ', num2str(theta_y)]);

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

xlabel('X');
ylabel('Y');
zlabel('Z');
view(3);

% % extrinsic rotation angles
% function R_z = R_yaw(angle)
%     R_z = [1, 0,           0;
%            0, cos(angle), -sin(angle);
%            0, sin(angle),  cos(angle)];
% end

% % extrinsic rotation angles
% function R_y = R_pitch(angle)
%     R_y = [cos(angle),  0, sin(angle);
%            0,           1, 0;
%           -sin(angle),  0, cos(angle)];
% end

% % extrinsic rotation angles
% function R_x = R_roll(angle)
%     R_x = [cos(angle), -sin(angle), 0;
%            sin(angle),  cos(angle), 0;
%            0,           0,          1];
% end