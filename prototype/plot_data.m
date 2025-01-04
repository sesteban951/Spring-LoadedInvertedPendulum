%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLot some sim data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load data
t = load('./data/time.csv');
x_com = load('./data/state_com.csv');
x_left = load('./data/state_left.csv');
x_right = load('./data/state_right.csv');
p_left = load('./data/pos_left.csv');
p_right = load('./data/pos_right.csv');
fileID = fopen('./data/domain.csv', 'r');
domain = textscan(fileID, '%s', 'Delimiter', ',');
domain = char(domain{1});
fclose(fileID);

% COM state
p_com = x_com(:,1:2);
v_com = x_com(:,3:4);

% leg states
q_left = x_left(:,1:2);
q_right = x_right(:,1:2);
v_left = x_left(:,3:4);
v_right = x_right(:,3:4);

% convert the domains to int
domain_int = zeros(length(domain), 1);
for i = 1:length(domain)
    if domain(i) == 'F'
        domain_int(i) = 0;
    elseif domain(i) == 'L'
        domain_int(i) = 1;
    elseif domain(i) == 'R'
        domain_int(i) = 2;
    elseif domain(i) == 'D'
        domain_int(i) = 3;
    end
end

% animation params
rt = 0.75; % realtime rate
plot_states = 0;
animate = 1;
replays = 3;

if plot_states == 1
    % plot all states
    figure('Name', 'COM States', 'WindowState', 'maximized');
    set(0, 'DefaultFigureRenderer', 'painters');

    subplot(3,4,1);
    hold on; grid on;
    plot(t, p_com(:,1), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$p_x$ [m]', 'Interpreter', 'latex');
    title('x-pos');

    subplot(3,4,2);
    hold on; grid on;
    plot(t, p_com(:,2), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$p_z$ [m]', 'Interpreter', 'latex');
    title('z-pos');

    subplot(3,4,5); 
    hold on; grid on;
    % vx_com = diff(p_com(:,1))./diff(t);
    plot(t, v_com(:,1), 'LineWidth', 2);
    % plot(t(1:end-1), vx_com, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$v_x$ [m/s]', 'Interpreter', 'latex');
    title('x-vel');

    subplot(3,4,6);
    hold on; grid on;
    % vz_com = diff(p_com(:,2))./diff(t);
    plot(t, v_com(:,2), 'LineWidth', 2);
    % plot(t(1:end-1), vz_com, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$v_z$ [m/s]', 'Interpreter', 'latex');
    title('z-vel');

    subplot(3,4,3);
    hold on; grid on;
    plot(t, q_left(:,1), 'LineWidth', 2);
    plot(t, q_right(:,1), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$r$ [m]', 'Interpreter', 'latex');
    legend('Left', 'Right');
    title('Leg Length, r');

    subplot(3,4,4);
    hold on; grid on;
    plot(t, q_left(:,2), 'LineWidth', 2);
    plot(t, q_right(:,2), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\theta$ [rad]', 'Interpreter', 'latex');
    legend('Left', 'Right');
    title('Leg Angle, theta');

    subplot(3,4,7);
    hold on; grid on;
    plot(t, v_left(:,1), 'LineWidth', 2);
    plot(t, v_right(:,1), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\dot{r}$ [m/s]', 'Interpreter', 'latex');
    legend('Left', 'Right');
    title('Leg Length Rate, r-dot');

    subplot(3,4,8);
    hold on; grid on;
    plot(t, v_left(:,2), 'LineWidth', 2);
    plot(t, v_right(:,2), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\dot{\theta}$ [rad/s]', 'Interpreter', 'latex');
    legend('Left', 'Right');
    title('Leg Angle Rate, theta-dot');
 
    subplot(3,4,[9:12]);
    hold on; grid on;
    stairs(t, domain_int, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('Domain');
    title('Domain');
    ylim([-0.5, 3.5]);
    yticks([0, 1, 2, 3]);
    yticklabels({'F', 'L', 'R', 'D'});
end

% animate the com trajectory
if animate == 1

    figure('Name', 'Animation');
    set(0, 'DefaultFigureRenderer', 'painters');
    hold on;

    xline(0);
    yline(0);
    xlabel('$p_x$ [m]', 'Interpreter', 'latex');
    ylabel('$p_z$ [m]', 'Interpreter', 'latex');
    grid on; axis equal;
    px_min = min([p_com(:,1); p_left(:,1); p_right(:,1)]);
    px_max = max([p_com(:,1); p_left(:,1); p_right(:,1)]);
    pz_min = min([p_com(:,2); p_left(:,2); p_right(:,2)]);
    pz_max = max([p_com(:,2); p_left(:,2); p_right(:,2)]);
    xlim([px_min-0.25, px_max+0.25]);
    ylim([min(0, pz_min)-0.25, pz_max+0.25]);
    
    t  = t * (1/rt);
   
    for i = 1:replays
        pause(0.25);
        tic;
        ind = 1;
        com_pts = [];
        left_foot_pts = [];
        right_foot_pts = [];
        while true

            % get COM position 
            px = p_com(ind,1);
            pz = p_com(ind,2);

            % draw the legs
            px_left = p_left(ind,1);
            pz_left = p_left(ind,2);
            px_right = p_right(ind,1);
            pz_right = p_right(ind,2);

            left_leg = plot([px, px_left], [pz, pz_left], 'b', 'LineWidth', 3);
            right_leg = plot([px, px_right], [pz, pz_right], 'r', 'LineWidth', 3);
            left_foot = plot(px_left, pz_left, 'bo', 'MarkerSize', 1, 'MarkerFaceColor', 'b');
            right_foot = plot(px_right, pz_right, 'ro', 'MarkerSize', 1, 'MarkerFaceColor', 'r');

            left_foot_pts = [left_foot_pts; left_foot];
            right_foot_pts = [right_foot_pts; right_foot];

            % draw the mass
            mass = plot(px, pz, 'ko', 'MarkerSize', 30, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
            pt_pos = plot(px, pz, 'k.', 'MarkerSize', 5);
            com_pts = [com_pts; pt_pos];

            drawnow;
            
            % title
            msg = sprintf('Time: %0.3f [sec]', t(ind) * rt);
            title(msg);
            
            % wait until the next time step
            while toc< t(ind+1)
                % wait
            end
            
            % increment the index
            if ind+1 >= length(t)
                break;
            else
                ind = ind + 1;
                delete(mass);
                delete(left_leg);
                delete(right_leg);
            end
        end

        % clean the plot if still replaying
        if i < replays
            delete(mass);
            delete(left_leg);
            delete(right_leg);
            for j = 1:length(com_pts)
                delete(com_pts(j));
                delete(left_foot_pts(j));
                delete(right_foot_pts(j));
            end
        end
    end
end