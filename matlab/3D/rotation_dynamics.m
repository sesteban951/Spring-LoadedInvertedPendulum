%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testing rotations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% sine wave function
A = (pi/2) * 1.0;  % WARNING: theta only in the range (-pi/2, pi/2)
f = 1;
phi = pi/2;
t = 0:0.01:3;
theta_x = A*sin(2*pi*f*t);
theta_y = A*sin(2*pi*f*t + phi);
thetadot_x = 2*pi*f*A*cos(2*pi*f*t);
thetadot_y = 2*pi*f*A*cos(2*pi*f*t + phi);

%  leg length
A_L = 0.2;
L = A_L*sin(2*pi*f*t) + 0.4;
Ldot = 2*pi*f*A_L*cos(2*pi*f*t);

% compute fwd kin
rt = zeros(3, length(t));
rt_dot = zeros(3, length(t));
for i = 1:length(t)
    [rt(:, i), rt_dot(:, i)] = fwd_kin(L(i), Ldot(i), theta_x(i), theta_y(i), thetadot_x(i), thetadot_y(i));
end

rt_x = rt(1,:);
rt_y = rt(2,:);
rt_z = rt(3,:);
rt_dot_x = rt_dot(1,:);
rt_dot_y = rt_dot(2,:);
rt_dot_z = rt_dot(3,:);

% plot the results
figure;
subplot(2, 3, 1);
hold on; grid on;
plot(t, rt_x,  'LineWidth', 2);
xlabel('time [s]');
ylabel('x [m]');

subplot(2, 3, 2);
hold on; grid on;
plot(t, rt_y, 'LineWidth', 2);
xlabel('time [s]');
ylabel('y [m]');

subplot(2, 3, 3);
hold on; grid on;
plot(t, rt_z, 'LineWidth', 2);
xlabel('time [s]');
ylabel('z [m]');

subplot(2, 3, 4);
hold on; grid on;
rt_dot_x_diff = diff(rt_x) ./ diff(t);
plot(t(1:end-1), rt_dot_x_diff, 'LineWidth', 2);
plot(t, rt_dot_x, 'LineWidth', 2);
xlabel('time [s]');
ylabel('xdot [m/s]');

subplot(2, 3, 5);
hold on; grid on;
rt_dot_y_diff = diff(rt_y) ./ diff(t);
plot(t(1:end-1), rt_dot_y_diff, 'LineWidth', 2);
plot(t, rt_dot_y, 'LineWidth', 2);
xlabel('time [s]');
ylabel('ydot [m/s]');

subplot(2, 3, 6);
hold on; grid on;
rt_dot_z_diff = diff(rt_z) ./ diff(t);
plot(t(1:end-1), rt_dot_z_diff, 'LineWidth', 2);
plot(t, rt_dot_z, 'LineWidth', 2);
xlabel('time [s]');
ylabel('zdot [m/s]');


theta = zeros(2, length(t));
thetadot = zeros(2, length(t));
for i = 1:length(t)
    [theta(:, i), thetadot(:, i)] = inv_kin(rt(:, i), rt_dot(:, i));
end

theta_x_calc = theta(1,:)+ 0.05;
theta_y_calc = theta(2,:)+ 0.05;
thetadot_x_calc = thetadot(1,:)+ 0.05;
thetadot_y_calc = thetadot(2,:)+ 0.05;

% plot the results
figure;
subplot(2, 2, 1);
hold on; grid on;
plot(t, theta_x,  'LineWidth', 2);
plot(t, theta_x_calc,  'LineWidth', 2);
xlabel('time [s]');
ylabel('theta_x [rad]');
legend('theta_x', 'theta_x calc');

subplot(2, 2, 2);
hold on; grid on;
plot(t, theta_y, 'LineWidth', 2);
plot(t, theta_y_calc, 'LineWidth', 2);
xlabel('time [s]');
ylabel('theta_y [rad]');
legend('theta_y', 'theta_y calc');

subplot(2, 2, 3);
hold on; grid on;
theta_x_diff = diff(theta_x) ./ diff(t);
plot(t(1:end-1), theta_x_diff, 'LineWidth', 2);
plot(t, thetadot_x, 'LineWidth', 2);
plot(t, thetadot_x_calc, 'LineWidth', 2);
xlabel('time [s]');
ylabel('thetadot_x [rad/s]');
legend('thetadot_x', 'thetadot_x calc');

subplot(2, 2, 4);
hold on; grid on;
theta_y_diff = diff(theta_y) ./ diff(t);
plot(t(1:end-1), theta_y_diff, 'LineWidth', 2);
plot(t, thetadot_y, 'LineWidth', 2);
plot(t, thetadot_y_calc, 'LineWidth', 2);
xlabel('time [s]');
ylabel('thetadot_y [rad/s]');
legend('thetadot_y', 'thetadot_y calc');

% forwad kinematics
function [r, rdot] = fwd_kin(L, Ldot, theta_x, theta_y, thetadot_x, thetadot_y)
    
    % position
    rz = -sqrt(L^2 / (tan(theta_x)^2 + tan(theta_y)^2 + 1));
    rx =  rz * tan(theta_y);
    ry = -rz * tan(theta_x);
    r = [rx, ry, rz]';
    
    %  velocity
    denom = 1 + tan(theta_x)^2 + tan(theta_y)^2;
    numer = tan(theta_x)*(sec(theta_x)^2)*thetadot_x + ...
            tan(theta_y)*(sec(theta_y)^2)*thetadot_y;
    rdot_z = -Ldot/sqrt(denom) + (L* numer) / (sqrt(denom^3));
    rdot_x =  rdot_z * tan(theta_y) + rz * sec(theta_y)^2 * thetadot_y;
    rdot_y = -rdot_z * tan(theta_x) - rz * sec(theta_x)^2 * thetadot_x;
    rdot = [rdot_x, rdot_y, rdot_z]';
end

% inverse kinematics
function [theta, thetadot] = inv_kin(r, rdot)
    
    % components the angles
    rx = r(1);
    ry = r(2);
    rz = r(3);
    theta_x = atan2(ry, -rz);
    theta_y = -atan2(rx, -rz);
    theta = [theta_x, theta_y];

    % components of the velocity
    rdot_x = rdot(1);
    rdot_y = rdot(2);
    rdot_z = rdot(3);
    thetadot_x = (ry * rdot_z - rz * rdot_y) / (rz^2 + ry^2);
    thetadot_y = (rz * rdot_x - rx * rdot_z) / (rz^2 + rx^2);
    thetadot = [thetadot_x, thetadot_y];
end
