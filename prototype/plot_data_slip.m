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
u_ankle = load('./data/slip/input_ankle.csv');
fileID = fopen('./data/slip/domain.csv', 'r');
domain = textscan(fileID, '%s', 'Delimiter', ',');
domain = char(domain{1});
fclose(fileID);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% segment the time
t_interval = [t(1) t(end)];
% t_interval = [0 1.0];

% plotting / animation
animate = 1;
rt = 1.0; % realtime rate
replays = 3;
save_video = 0;
plot_com = 0;
plot_foot = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx = find(t >= t_interval(1) & t <= t_interval(2));
t = t(idx);
x_sys = x_sys(idx,:);
x_leg = x_leg(idx,:);
x_foot = x_foot(idx,:);
lambd = lambd(idx,:);
u = u(idx,:);
domain = domain(idx,:);

% system state
p_com = x_sys(:,1:2);
v_com = x_sys(:,3:4);
leg_pos_commands = x_sys(:,5:6);

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
    plot(t, v_com(:,1), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$v_x$ [m/s]', 'Interpreter', 'latex');
    title('x-vel');

    subplot(3,6,8);
    hold on; grid on;
    plot(t, v_com(:,2), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$v_z$ [m/s]', 'Interpreter', 'latex');
    title('z-vel');

    % LEG STATES
    subplot(3,6,3);
    hold on; grid on;
    plot(t, x_leg(:,1), 'LineWidth', 2);
    plot(t, leg_pos_commands(:,1), 'LineWidth', 1.0);
    xlabel('Time [sec]');
    ylabel('$r$ [m]', 'Interpreter', 'latex');
    title('Leg Length, r');
    legend('actual', 'command');

    subplot(3,6,4);
    hold on; grid on;
    plot(t, x_leg(:,2), 'LineWidth', 2);
    plot(t, leg_pos_commands(:,2), 'LineWidth', 1.0);
    xlabel('Time [sec]');
    ylabel('$\theta$ [rad]', 'Interpreter', 'latex');
    title('Leg Angle, theta');
    legend('actual', 'command');

    subplot(3,6,9);
    hold on; grid on;
    plot(t, x_leg(:,3), 'LineWidth', 2);
    plot(t, u(:,1), 'LineWidth', 1.0);
    xlabel('Time [sec]');
    ylabel('$\dot{r}$ [m/s]', 'Interpreter', 'latex');
    title('Leg Length Rate, r-dot');
    legend('actual', 'command');

    subplot(3,6,10);
    hold on; grid on;
    plot(t, x_leg(:,4), 'LineWidth', 2);
    plot(t, u(:,2), 'LineWidth', 1.0);
    xlabel('Time [sec]');
    ylabel('$\dot{\theta}$ [rad/s]', 'Interpreter', 'latex');
    title('Leg Angle Rate, theta-dot');
    legend('actual', 'command');

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
    subplot(3,6,13);
    plot(t, lambda_mag, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\lambda$ [N]', 'Interpreter', 'latex');
    title('Lambda Magnitude');
    grid on;

    % ANKLE
    subplot(3,6,14);
    plot(t, u_ankle, 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('$\tau$ [Nm]', 'Interpreter', 'latex');
    title('Torque Ankle');
    grid on;

    % INPUT
    subplot(3,6,[15,16]); 
    hold on; grid on;
    plot(t, u(:,1), 'LineWidth', 2);
    plot(t, u(:,2), 'LineWidth', 2);
    xlabel('Time [sec]');
    ylabel('Input');
    title('l0 rate input');
    legend('$l_0$', '$\theta$', 'interpreter', 'latex');
    grid on;

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
    
    % Video writer setup
    if save_video == 1
        video_filename = 'SLIP_CEM';  % Set your video file name
        v = VideoWriter(video_filename, 'Motion JPEG AVI');  % Create a video writer object
        v.FrameRate = 1/rt;  % Adjust the frame rate based on your animation speed
        open(v);  % Open the video file for writing
    end

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
            leg = plot([px, px_foot], [pz, pz_foot], 'k', 'LineWidth', 3);
            ball_foot = plot(px_foot, pz_foot, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k');
            
            % draw the mass
            if domain(ind) == 'F'
                mass = plot(px, pz, 'ko', 'MarkerSize', 35, 'MarkerFaceColor', [0 0.4470 0.7410], 'LineWidth', 1.5, 'MarkerEdgeColor', 'k');
            elseif domain(ind) == 'G'
                mass = plot(px, pz, 'ko', 'MarkerSize', 35, 'MarkerFaceColor', [0.6350 0.0780 0.1840], 'LineWidth', 1.5, 'MarkerEdgeColor', 'k');
            end

            %  draw trajectory trail
            if plot_foot == 1
                foot = plot(px_foot, pz_foot, 'bo', 'MarkerSize', 1, 'MarkerFaceColor', 'b');
                foot_pts = [foot_pts; foot];
            end
            if plot_com ==1
                pt_pos = plot(px, pz, 'k.', 'MarkerSize', 5);
                com_pts = [com_pts; pt_pos];
            end

            % Capture the frame and write to video
            if save_video == 1
                frame = getframe(gcf);  % Capture current figure frame
                writeVideo(v, frame);   % Write frame to video
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
                delete(ball_foot);
            end
        end

        % Close video writer
        if save_video == 1
            close(v);
            break;
        end

        % clean the plot if still replaying
        if i < replays
            delete(mass);
            delete(leg);
            delete(ball_foot);
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