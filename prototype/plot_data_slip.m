%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot some sim data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load data
t = load('./data/slip/time.csv');
x_sys = load('./data/slip/state_com.csv');
x_leg = load('./data/slip/state_leg.csv');
x_foot = load('./data/slip/state_foot.csv');
lambd = load('./data/slip/lambd.csv');
u = load('./data/slip/input.csv');
fileID = fopen('./data/slip/domain.csv', 'r');
domain = textscan(fileID, '%s', 'Delimiter', ',');
domain = char(domain{1});
fclose(fileID);

% system state
p_com = x_sys(:,1:2);
v_com = x_sys(:,3:4);
q_leg = x_sys(:,5:6);

% leg states
r = x_leg(:,1);
theta = x_leg(:,2);
rdot = x_leg(:,3);
thetadot = x_leg(:,4);

% foot states
p_foot = x_foot(:,1:2);
v_foot = x_foot(:,3:4);

% compute lambda magnitude
lambda_mag = zeros(length(t), 1);
for i = 1:length(t)
    lam = lambd(i,1:2);
    lambda_mag(i) = norm(lam, 2);
end

% convert the domains to int
domain_int = zeros(length(domain),1);
for i = 1:length(domain)
    if domain(i) == 'F'
        domain_int(i) = 1;
    elseif domain(i) == 'G'
        domain_int(i) = 0;
    end
end

animate = 0;
rt = 0.05; % realtime rate
replays = 1;
plot_com = 0;
plot_foot = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if animate == 0
    % plot all states
    figure('Name', 'COM States', 'WindowState', 'maximized');

    % COM STATES
    subplot(3,6,1);
    hold on; grid on;
    plot(t, p_com(:,1), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$p_x$ [m]', 'Interpreter', 'latex');
    title('x-pos');

    subplot(3,6,2);
    hold on; grid on;
    plot(t, p_com(:,2), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$p_z$ [m]', 'Interpreter', 'latex');
    title('z-pos');

    subplot(3,6,7); 
    hold on; grid on;
    % vx_sys = diff(p_com(:,1))./diff(t);
    plot(t, v_com(:,1), 'LineWidth', 2);
    % plot(t(1:end-1), vx_sys, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$v_x$ [m/s]', 'Interpreter', 'latex');
    title('x-vel');

    subplot(3,6,8);
    hold on; grid on;
    % vz_com = diff(p_com(:,2))./diff(t);
    plot(t, v_com(:,2), 'LineWidth', 2);
    % plot(t(1:end-1), vz_com, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$v_z$ [m/s]', 'Interpreter', 'latex');
    title('z-vel');

    % LEG STATES
    subplot(3,6,3);
    hold on; grid on;
    plot(t, q_leg(:,1), 'LineWidth', 1.5);
    plot(t, r, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$r$ [m]', 'Interpreter', 'latex');
    title('Leg Length, r');

    subplot(3,6,4);
    hold on; grid on;
    plot(t, q_leg(:,2), 'LineWidth', 1.5);
    plot(t, theta, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\theta$ [rad]', 'Interpreter', 'latex');
    title('Leg Angle, theta');

    subplot(3,6,9);
    hold on; grid on;
    plot(t, rdot, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\dot{r}$ [m/s]', 'Interpreter', 'latex');
    title('Leg Length Rate, r-dot');

    subplot(3,6,10);
    hold on; grid on;
    plot(t, thetadot, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\dot{\theta}$ [rad/s]', 'Interpreter', 'latex');
    title('Leg Angle Rate, theta-dot');

    % FOOT STATES
    subplot(3,6,5);
    plot(t, p_foot(:,1), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$p_{foot,x}$ [m]', 'Interpreter', 'latex');
    title('Foot x-pos');
    grid on;

    subplot(3,6,6);
    plot(t, p_foot(:,2), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$p_{foot,z}$ [m]', 'Interpreter', 'latex');
    title('Foot z-pos');
    grid on;

    subplot(3,6,11);
    plot(t, v_foot(:,1), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$v_{foot,x}$ [m/s]', 'Interpreter', 'latex');
    title('Foot x-vel');
    grid on;

    subplot(3,6,12);
    plot(t, v_foot(:,2), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$v_{foot,z}$ [m/s]', 'Interpreter', 'latex');
    title('Foot z-vel');
    grid on;

    % LAMBDA
    subplot(3,6,[13,14]);
    plot(t, lambda_mag, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\lambda$ [N]', 'Interpreter', 'latex');
    title('Lambda Magnitude');
    grid on;

    % % INPUT
    % subplot(3,6,[15,16]); 
    % hold on; grid on;
    % plot(t(1:end-1), u(:,1), 'LineWidth', 2);
    % plot(t(1:end-1), u(:,2), 'LineWidth', 2);
    % xlabel('Time [sec]');
    % ylabel('Input');
    % title('l0 rate input');
    % grid on;

    % DOMAIN
    subplot(3,6,[17:18]);
    hold on; grid on;
    stairs(t, domain_int, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('Domain');
    title('Domain');
    ylim([-0.5, 3.5]);
    yticks([0, 1]);
    yticklabels({'G', 'F'});

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% animate the com trajectory
if animate == 1

    figure('Name', 'Animation');
    hold on;

    xline(0);
    yline(0);
    xlabel('$p_x$ [m]', 'Interpreter', 'latex');
    ylabel('$p_z$ [m]', 'Interpreter', 'latex');
    grid on; axis equal;
    px_min = min([p_com(:,1); p_foot(:,1)]);
    px_max = max([p_com(:,1); p_foot(:,1)]);
    pz_min = min([p_com(:,2); p_foot(:,2)]);
    pz_max = max([p_com(:,2); p_foot(:,2)]);
    xlim([px_min-0.25, px_max+0.25]);
    ylim([min(0, pz_min)-0.25, pz_max+0.25]);
    
    t  = t * (1/rt);
   
    for i = 1:replays
        pause(0.25);
        tic;
        ind = 1;
        com_pts = [];
        foot_pts = [];
        while true

            % get COM position 
            px = p_com(ind,1);
            pz = p_com(ind,2);

            % draw the legs
            px_foot = p_foot(ind,1);
            pz_foot = p_foot(ind,2);
            if domain(ind) == 'F'
                leg = plot([px, px_foot], [pz, pz_foot], 'b', 'LineWidth', 3);
            elseif domain(ind) == 'G'
                leg = plot([px, px_foot], [pz, pz_foot], 'r', 'LineWidth', 3);
            end
            if plot_foot == 1
                foot = plot(px_foot, pz_foot, 'bo', 'MarkerSize', 1, 'MarkerFaceColor', 'b');
                foot_pts = [foot_pts; foot];
            end

            % draw the mass
            mass = plot(px, pz, 'ko', 'MarkerSize', 30, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
            if plot_com ==1
                pt_pos = plot(px, pz, 'k.', 'MarkerSize', 5);
                com_pts = [com_pts; pt_pos];
            end

            drawnow;
            
            % title
            msg = sprintf('Time: %0.3f [sec]\n vx = %0.3f, px = %0.3f\n vz = %0.3f, pz = %0.3f',...
                         t(ind) * rt, v_com(ind,1), p_com(ind,1), v_com(ind,2), p_com(ind,2));
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
                delete(leg);
            end
        end

        % clean the plot if still replaying
        if i < replays
            delete(mass);
            delete(leg);
            for j = 1:length(com_pts)
                if plot_com == 1
                    delete(com_pts(j));
                end
                if plot_foot == 1
                    delete(foot_pts(j));
                end
            end
        end
    end
end