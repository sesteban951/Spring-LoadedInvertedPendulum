%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT SOME SIM RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load data
x_sys = load('../data/state_sys.csv');
x_leg = load('../data/state_leg.csv');
x_foot = load('../data/state_foot.csv');
u = load('../data/input.csv');
d = load('../data/domain.csv');

figure;
plot(x_sys(:,1), x_sys(:,2), 'r');
figure;
plot(x_leg(:,1), 'r');
figure;
plot(x_leg(:,2), 'r');
figure;
plot(x_foot(:,1), x_foot(:,2), 'r');
