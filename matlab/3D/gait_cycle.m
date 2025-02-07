%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing Gait Cycles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% gait cycle parameters
T_gait = 2.0;
r_L = 0.5;
r_R = 0.5;
offset_L = 0.25;
offset_R = 0.0;

% time vector
T_sim = 9.0;
t = 0:0.01:T_sim;

% gait cycle 
signal_L = zeros(1,length(t));
signal_R = zeros(1,length(t));

% gait cycle signal
for i = 1:length(t)
    % left leg
    if mod(t(i) - offset_L * T_gait, T_gait) < r_L*T_gait
        signal_L(i) = 1;
    else 
        signal_L(i) = 0;
    end

    % right leg
    if mod(t(i) - offset_R * T_gait, T_gait) < r_R*T_gait
        signal_R(i) = 1;
    else 
        signal_R(i) = 0;
    end
end

% plot
figure();

% subplot(1,2,1)
grid on; hold on;
plot(t, signal_L, 'b', 'LineWidth', 2.2)
plot(t, signal_R, 'r', 'LineWidth', 1.5)
xlabel('Time (s)')
ylabel('Contact')
title('Gait Cycle')
xlim([0 T_sim])
ylim([-0.1 1.1])
yticks([0 1])

