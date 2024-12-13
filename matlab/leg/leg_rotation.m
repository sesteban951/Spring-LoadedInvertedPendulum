clear all; close all; clc;

% parameters
r = 1.0;

% roll and pitch angles
phi = 35.0;
psi = -60.0;

% zero position 
l_0 = [0; 0; -r];

% individual rotation matrices
phi = phi * pi/180;
psi = psi * pi/180;
R_phi = [1, 0, 0;
         0, cos(phi), -sin(phi);
         0, sin(phi), cos(phi)];
R_psi = [cos(psi),  0, sin(psi);
         0,         1, 0;
         -sin(psi), 0, cos(psi)];

% leg position after the rotation
% l_f = R_phi * R_psi * l_0
l_f = R_psi * R_phi * l_0

% plot the intial and final leg positions
figure;
grid on; hold on; 
view(3);
axis equal;

% draw the x, y, z axes as arrows
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);

% draw the leg
plot3([0, 0], [0, 0], [0, 0], 'k.', 'MarkerSize', 20);
plot3([0, 0], [0, 0], [0, -r], 'k--', 'LineWidth', 2);
plot3([0, l_f(1)], [0, l_f(2)], [0, l_f(3)], 'k', 'LineWidth', 2);
plot3(l_f(1), l_f(2), l_f(3), 'k.', 'MarkerSize', 20);

xlabel('x'); 
ylabel('y'); 
zlabel('z');

% set the title with the angle values
msg = sprintf('phi = %.1f deg, psi = %.1f deg', phi*180/pi, psi*180/pi);
title(msg);