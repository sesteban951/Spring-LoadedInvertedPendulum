%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing Gait Cycles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% gait cycle parameters
% T_gait = 1.0;
% c_L = 0.5;
% c_R = 0.5;
% offset_L = 0.0;
% offset_R = 0.5;

% based on velocity
v_des = 0.75;
[T_gait, c_L, c_R, offset_L, offset_R] = gait_cycle_parameters(v_des)

% time vector
T_sim = 10.0;
t = 0:0.01:T_sim;

% gait cycle 
signal_L = zeros(1,length(t));
signal_R = zeros(1,length(t));

% gait cycle signal
for i = 1:length(t)
    % left leg
    if mod(t(i)/T_gait - offset_L, 1.0) < c_L
        signal_L(i) = 1;
    else 
        signal_L(i) = 0;
    end

    % right leg
    if mod(t(i)/T_gait - offset_R, 1.0) < c_R
        signal_R(i) = 1;
    else 
        signal_R(i) = 0;
    end
end

% plot
figure('Name', 'Gait Cycle', 'WindowState', 'maximized');

subplot(1,2,1)
grid on; hold on;
plot(t, signal_L, 'b', 'LineWidth', 2.2)
plot(t, signal_R, 'r', 'LineWidth', 1.5)
xlabel('Time (s)')
ylabel('Contact')
title('Gait Cycle')
xlim([0 T_sim])
ylim([-0.1 1.1])
yticks([0 1])
n = floor(T_sim/T_gait);
for i = 1:n
    xline(i*T_gait, '--k', 'LineWidth', 1.5)
end

pause(0.1);

subplot(1,2,2)

pz_com = 0.7;
px_foot = 0.1;
pz_foot = 0.1;

hold on; grid on; axis equal;
xline(0); yline(0);
xlim([-px_foot - 0.1, px_foot + 0.1]); 
ylim([-0.1 pz_com + 0.1]);

plot(0, 0.7, 'ko', 'MarkerSize', 30, 'MarkerFaceColor', 'k');

idx = 1;
tic;
while idx < length(t)
    t_now = t(idx);

    % left leg
    L = signal_L(idx);
    R = signal_R(idx);
    if L ==1
        foot_L = plot(px_foot, 0, 'bo', 'MarkerSize', 15, 'MarkerFaceColor', 'b');
        leg_L = plot([0, px_foot], [pz_com, 0], 'b', 'LineWidth', 2.5);
    elseif L == 0
        foot_L = plot(px_foot, pz_foot, 'bo', 'MarkerSize', 10);
        leg_L = plot([0, px_foot], [pz_com, pz_foot], 'b', 'LineWidth', 1.5);
    end
    if R ==1
        foot_R = plot(-px_foot, 0, 'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
        leg_R = plot([0, -px_foot], [pz_com, 0], 'r', 'LineWidth', 2.5);
    elseif R == 0
        foot_R = plot(-px_foot, pz_foot, 'ro', 'MarkerSize', 10);
        leg_R = plot([0, -px_foot], [pz_com, pz_foot], 'r', 'LineWidth', 1.5);
    end

    drawnow;

    idx = idx + 1;

    msg = sprintf('Time: %.2f s', t_now);
    title(msg);

    while toc< t(idx)
        % do nothing
    end

    if idx < length(t)
        delete(foot_L);
        delete(foot_R);
        delete(leg_L);
        delete(leg_R);
    end
end

% gait cycle parameters as a function of velocity
function [T_gait, c_L, c_R, offset_L, offset_R] = gait_cycle_parameters(v_des)
    
    % assume that the gait cycle is perfectly out of phase
    offset_L = 0.0;
    offset_R = 0.5;

    % fix low and high bounds
    c_min = 0.2;
    c_max = 1.0;
    Tg_min = 0.5;
    Tg_max = 1.2;
    v_max = 1.5;

    % compute contact ratio
    c_L = 1 + (c_min-1)/ (v_max) * v_des;
    c_R = 1 + (c_min-1)/ (v_max) * v_des;

    % compute the gait cycle period
    T_gait = Tg_min + (Tg_max - Tg_min) / (v_max) * v_des;

end