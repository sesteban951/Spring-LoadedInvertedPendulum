%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For looking at SLIP data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear all;

% Load data
sol_jax = readmatrix('./data/solution_jax.csv');
sol_jax_batched = readmatrix('./data/solution_jax_batch.csv');
sol_no_jax = readmatrix('./data/solution.csv');

% plot data
lineWidth = 0.5;

figure(1); hold on;
plot(sol_no_jax(:,1), sol_no_jax(:,2), 'r', 'LineWidth', lineWidth)
plot(sol_jax(:,1), sol_jax(:,2), 'b', 'LineWidth', lineWidth)
plot(sol_jax_batched(:,1), sol_jax_batched(:,2), 'g', 'LineWidth', lineWidth)
xlabel('x1')
ylabel('x2')
legend('no jax', 'jax', 'jax batched')



