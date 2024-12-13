clear all; close all; clc;

% import data
data_newton = csvread('data/newton.csv');
data_lagrange = csvread('data/lagrange.csv');

% plot the data
t_newton = data_newton(:, 1);
x_newton = data_newton(:, 2:end);
t_lagrange = data_lagrange(:, 1);
x_lagrange = data_lagrange(:, 2:end);

% plot the data
figure;
subplot(2, 2, 1);
hold on; 
plot(t_newton, x_newton(:, 1), 'b', 'LineWidth', 1);
plot(t_lagrange, x_lagrange(:, 1), 'r', 'LineWidth', 1);
xlabel('t'); ylabel('px');
legend('Newton', 'Lagrange');

subplot(2, 2, 2);
hold on;
plot(t_newton, x_newton(:, 2), 'b', 'LineWidth', 1);
plot(t_lagrange, x_lagrange(:, 2), 'r', 'LineWidth', 1);
xlabel('t'); ylabel('pz');
legend('Newton', 'Lagrange');

subplot(2, 2, 3);
hold on;
plot(t_newton, x_newton(:, 3), 'b', 'LineWidth', 1);
plot(t_lagrange, x_lagrange(:, 3), 'r', 'LineWidth', 1);
xlabel('t'); ylabel('px dot');
legend('Newton', 'Lagrange');

subplot(2, 2, 4);
hold on;
plot(t_newton, x_newton(:, 4), 'b', 'LineWidth', 1);
plot(t_lagrange, x_lagrange(:, 4), 'r', 'LineWidth', 1);
xlabel('t'); ylabel('pz dot');
legend('Newton', 'Lagrange');
