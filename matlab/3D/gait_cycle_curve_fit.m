%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% emperical data from gait cycle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% data container
data = [];

% 0.125 m/s
v_des = 0.125;
up_times = [4.93 5.66  6.41 6.92 7.84 8.45];
down_times = [5.53 6.25 6.75 7.63 8.3 9.11];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 0.25 m/s
v_des = 0.25;
up_times = [4.58 5.3 6.14 6.94 7.71];
down_times = [5 5.83 6.62 7.39 8.27];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 0.375 m/s
v_des = 0.375;
up_times = [5.18 5.91 6.67 7.47 8.25 9.13];
down_times = [5.58 6.28 7.1 7.86 8.69 9.48];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 0.50 m/s
v_des = 0.5;
up_times = [5.9 6.53 7.15 7.91 8.56];
down_times = [6.27 6.82 7.48 8.26 8.88];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 0.625 m/s
v_des = 0.625;
up_times = [4.92 5.61 6.22 6.89 7.45 8.05];
down_times = [5.19 5.87 6.5 7.15 7.73 8.28];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 0.75 m/s
v_des = 0.75;
up_times = [1.85 2.42 3.03 3.65 4.27];
down_times = [2.09 2.68 3.3 3.9 4.53];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 0.875 m/s
v_des = 0.875;
up_times = [5.51 6.1 6.75 7.37 7.95 8.53];
down_times = [5.75 6.36 7.01 7.62 8.19 8.78];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 1.00 m/s
v_des = 1.0;
up_times = [3.28 3.9 4.48 5.1 5.7];
down_times = [3.52 4.12 4.72 5.34 5.94];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 1.25 m/s
v_des = 1.25;
up_times = [5.22 5.78 6.37 6.98 7.56];
down_times = [5.44 5.99 6.59 7.21 7.79];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 1.5 m/s
v_des = 1.5;
up_times = [5.4 6.03 6.61 7.22 7.81];
down_times = [5.61 6.24 6.81 7.43 8.01];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 1.75 m/s
v_des = 1.75;
up_times = [5.31 6.03 6.69 7.33 7.99];
down_times = [5.52 6.23 6.89 7.53 8.2];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% 2.0 m/s
v_des = 2.0;
up_times = [4.67 5.45 6.16 6.88 7.68];
down_times = [4.87 5.64 6.35 7.1 7.89];
[v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des);
data = [data; v_des, T_cycle, c];

% add the standing data
% data = [0, 0.8, 1.075; data];

% curve fit a quadratic c sv v_des
p_T = polyfit(data(:,1), data(:,2), 2)
p_c = polyfit(data(:,1), data(:,3), 3)

% display the curve fit parameters
disp("*******************************************************")
disp("T_cycle = (" + p_T(1) + ")*v^2  + (" + p_T(2) + ")*v + (" + p_T(3)+ ")");
disp("      c = (" + p_c(1) + ")*v^3 + (" + p_c(2) + ")*v^2 + (" + p_c(3) + ")*v + (" + p_c(4) + ")");
disp("*******************************************************")

% build the 3D plot
figure;
subplot(2,1,1);
hold on;
plot(data(:,1), data(:,2), 'o-');
plot(data(:,1), polyval(p_T, data(:,1)), 'r-');
xlabel('v_{des} [m/s]');
ylabel('T_{cycle} [s]');
title('Gait Cycle Period vs. Desired Velocity');
grid on;

subplot(2,1,2);
hold on;
plot(data(:,1), data(:,3), 'o-');
plot(data(:,1), polyval(p_c, data(:,1)), 'r-');
xlabel('v_{des} [m/s]');
ylabel('c');
title('Contact Ratio vs. Desired Velocity');
grid on;


% function to compute the gait parameters
function [v_des, T_cycle, c] = copmute_gait_params(up_times, down_times, v_des)

    % compute the periods
    T_up = diff(up_times);
    T_down = diff(down_times);

    % compute the average gait cycle
    T_cycle = mean([T_up, T_down]);

    % compute the contact time
    T_contact = down_times - up_times;
    T_contact_avg = mean(T_contact);

    T_no_contact = up_times(2:end) - down_times(1:end-1); 
    T_no_contact_avg = mean(T_no_contact);

    % compute the contact average
    contact = T_contact_avg / T_cycle;
    no_contact = (T_cycle - T_no_contact_avg) / T_cycle;
    c = mean([contact, no_contact]);

    % display some results
    disp("----------------------------")
    disp("v_des = " + v_des + " [m/s]")
    disp("T_cycle = " + T_cycle + " [s]")
    disp("c = " + c)

end