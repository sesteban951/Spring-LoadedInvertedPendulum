%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spinrg Loaded Inverted Pendulum (SLIP)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% SLIP params
params.m = 22;           % CoM mass (Achilles mass 22 kg)
params.g = 9.81;         % gravity
params.l0 = 0.5;         % spring free length (Achilles leg length 0.7 m)
params.K = [0.03, 0.18]; % Raibert controller gain
alpha_max_deg = 60;      % max foot angle from verticle [deg]
params.alpha_max = alpha_max_deg * (pi/180);  % max foot angle [rad]

% trajectory parameters
params.traj_type = "vel"; % "pos" for constant velocity and "vel" for constant position 
params.v_des = 1.5;       % converge to a fixed velocity
params.p_des = 1.0;       % converge to a fixed position

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

animation = 1;     % full SLIP replay
save_video = 1;    % save the video

% plotting parameters
realtime_rate = 1.0;
n_points = 45;
max_num_transitions = 40;

dom = 0;   % plot ground and flight phases
orbit = 0;    % plot orbit plots
poincare = 0; % plot the poincare map results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sim params
freq = 150;
dt = 1/freq;
tspan = 0:dt:3.0;  % to allow for switching before timeout

% initial conditions (always start in flight)
x0 = [0.0;   % x
      1.25;   % z
      0.5;   % x_dot
      0.0];  % z_dot
domain = "flight";

% initial foot angle
alpha_LO = 0;
alpha = angle_control(0.0, x0, params);

% set the switching manifolds
options_g2f = odeset('Events', @(t,x)ground_to_flight(t, x, params), 'RelTol', 1e-8, 'AbsTol', 1e-9);
options_poincare = odeset('Events', @(t,x)poincare_section(t, x), 'RelTol', 1e-8, 'AbsTol', 1e-9);

% simulate the hybrid system
t_current = 0;
num_transitions = 0;
D = [];  % domain storage
T = [];  % time storage
X = [];  % state storage
F = [];  % ground foot position storage
T_apex = [];  % apex time storage
X_apex = [];  % apex state storage
T_TD = [];  % touch down times
T_LO = [];  % lift off times

tic;
while num_transitions <= max_num_transitions
    
    % switch domains
    if domain == "flight"

        % set the options with the appropitate alpha
        options_f2g = odeset('Events', @(t,x)flight_to_ground(t, x, alpha, params), 'RelTol', 1e-8, 'AbsTol', 1e-9);

        % flight: x = [x, z, x_dot, z_dot]
        [t_flight, x_flight] = ode45(@(t,x)dynamics_f(t,x,params), tspan + t_current, x0, options_f2g);
        [~, ~, t_apex, x_apex, ~] = ode45(@(t,x)dynamics_f(t,x,params), tspan + t_current, x0, options_poincare); % purely used for poincare section
        T_apex = [T_apex; t_apex];
        X_apex = [X_apex; x_apex];

        % store the trajectory
        D = [D; 0 * ones(length(t_flight),1)];
        T = [T; t_flight];
        X = [X; x_flight];
        T_TD = [T_TD; t_flight(end)];
  
        % udpate the current time and the intial state
        t_current = T(end);

        % compute foot trajectory
        for i = 1:length(t_flight)
            beta = (t_flight(i) - t_flight(1)) / (t_flight(end) - t_flight(1));
            alpha_ = (1 - beta) * alpha_LO + beta * alpha;
            p_foot = [x_flight(i,1) + params.l0 * sin(alpha_); 
                      x_flight(i,2) - params.l0 * cos(alpha_)];
            F = [F; p_foot'];
        end

        % set new initial condition
        x0 = x_flight(end,:);
        x0 = cart_to_polar(x0, params, alpha);

        % define new domain
        domain = "ground";
        num_transitions = num_transitions + 1;

    elseif domain == "ground"
        
        % ground: x = [r, theta, r_dot, theta_dot]
        [t_ground, x_ground] = ode45(@(t,x)dynamics_g(t,x,params), tspan + t_current, x0, options_g2f); 

        % save the lift off angle
        alpha_LO = -x_ground(end,2);

        % convert the polar state to cartesian
        for i = 1:length(t_ground)
            x_ground(i,:) = polar_to_cart(x_ground(i,:)); % convert it to cartesian
            x_ground(i,1) = x_ground(i,1) + p_foot(1);    % add the foot position offset
        end

        % store the trajectory
        D = [D; 1 * ones(length(t_ground),1)];
        T = [T; t_ground];
        X = [X; x_ground];
        T_LO = [T_LO; t_ground(end)];

        % udpate the current time and the intial state
        t_current = T(end);

        % compute foot trajectory
        for i = 1:length(t_ground)
            p_foot = F(end,:);
            F = [F; p_foot];
        end

        % set new initial condition
        x0 = x_ground(end,:);
        alpha = angle_control(t_current, x0, params);
    
        % define new domain
        domain = "flight";
        num_transitions = num_transitions + 1;
    end
end
msg = "Time to simulate the SLIP model: " + string(toc) + " seconds";
disp(msg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if dom == 1

    % plot the domain
    figure("Name", "Domain Plot");
    hold on; grid on;

    % plot the domain times
    subplot(2,1,1)
    area(T, -D+1, 'FaceColor', 'b', 'LineWidth', 2, 'FaceAlpha', 0.5);
    title("Domain Plot", 'FontSize', 16);
    xlabel('Time [s]', 'FontSize', 16);
    ylabel('Domain', 'FontSize', 16);
    yticks([0, 1]);

    % touch down and lift off times
    subplot(2,1,2)
    hold on;
    plot(T_TD, 'bx', 'MarkerSize', 16);
    plot(T_LO, 'rx', 'MarkerSize', 16);
    ylabel('TD / LO Times [s]', 'FontSize', 16);
    xlim([0, T(end)]);
    legend('Touch Down', 'Lift Off', 'FontSize', 16, 'Location', 'best');
    grid on;
end

if orbit == 1
    figure("Name", "Orbit Diagram");
    hold on; grid on;

    % unpack the state
    plot(X(:,2), X(:,4), 'b', 'MarkerSize', 10, 'LineWidth', 3.0);
    xline(0); yline(0);
    title("Orbit Plot", 'FontSize', 16);
    xlabel('$z$ [m]', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('$\dot{z}$ [m/s]', 'Interpreter', 'latex', 'FontSize', 16);
end

if poincare == 1
    
    figure("Name", "Poincare Section");
    hold on; grid on;

    % plot the poincare section
    plot(X(:,2), X(:,3), 'k--', 'MarkerSize', 5);
    plot(X_apex(:,2), X_apex(:,3), 'kx', 'MarkerSize', 15);
    xline(0); yline(0);
    title("Apex-to-Apex", 'FontSize', 16);
    xlabel('$z$ [m]', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('$\dot{x}$ [m/s]', 'Interpreter', 'latex', 'FontSize', 16);
end

if animation == 1
    % create a new figure
    figure('Name', 'SLIP Simulation');
    hold on;
    xline(0); yline(0);
    xlabel('$p_x$ [m]', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('$p_z$ [m]', 'Interpreter', 'latex', 'FontSize', 16);
    axis equal;

    % set axis limits
    z_min = -0.1;
    z_max = max(X(:,2)) + 0.1;
    ylim([z_min, z_max]);

    % draw the desired trajectory
    if params.traj_type == "pos"
        target = xline(params.p_des, '--', 'Target', 'Color', "k", 'LineWidth', 1.5);
    end

    % scale time for animation
    T = T / realtime_rate;

    % Number of points to record
    com_history = [];      % Initialize an empty array to store the last 20 COM points
    com_history_plot = []; % Initialize a variable to store the plot handle for the history

    % Video writer setup
    if save_video == 1
        video_filename = 'slip_simulation_video.mp4';  % Set your video file name
        v = VideoWriter(video_filename, 'Motion JPEG AVI');  % Create a video writer object
        v.FrameRate = 1/realtime_rate;  % Adjust the frame rate based on your animation speed
        open(v);  % Open the video file for writing
    end

    tic;
    t_now = T(1);
    ind = 1;
    while t_now < T(end)

        % plot the foot and pole
        pole = plot([F(ind,1), X(ind,1)], [F(ind,2), X(ind,2)], 'k', 'LineWidth', 2.5);
        foot = plot(F(ind,1), F(ind,2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');  % in flight

        % plot the SLIP COM
        if D(ind) == 0
            com = plot(X(ind,1), X(ind,2), 'ko', 'MarkerSize', 30, 'MarkerFaceColor', 'b');  % on the ground
        elseif D(ind) == 1
            com = plot(X(ind,1), X(ind,2), 'ko', 'MarkerSize', 30, 'MarkerFaceColor', 'r');  % in flight
        end

        % Update the history of the last n_points
        com_history = [com_history; X(ind, [1, 2])];  % Add the current point to history
        if size(com_history, 1) > n_points
            com_history = com_history(2:end, :);  % Remove the oldest point if we exceed n_points
        end

        % Clear previous history plot
        if ~isempty(com_history_plot)
            delete(com_history_plot);
        end

        % Plot the last 20 points
        com_history_plot = plot(com_history(:, 1), com_history(:, 2), 'g.', 'MarkerSize', 8);  % Plot history
        
        % current time
        plot_title = sprintf("Time: %.2f\n apex = [%.2f, %.2f]", T(ind) * realtime_rate, X(ind,2), X(ind,3));
        title(plot_title,'FontSize', 14);   
        
        % adjust the x_axis width
        x_min = X(ind,1) - 1.0;
        x_max = X(ind,1) + 1.0;
        xlim([x_min, x_max]);

        % Capture the frame and write to video
        if save_video == 1
            frame = getframe(gcf);  % Capture current figure frame
            writeVideo(v, frame);   % Write frame to video
        end

        drawnow;

        % wait until the next time step
        while toc < T(ind+1)
            % wait
        end

        % increment the index
        if ind+1 == length(T)
            break;
        else
            ind = ind + 1;
            delete(pole);
            delete(foot);
            delete(com);
        end
    end

    % Close video writer
    if save_video == 1
        close(v);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMICS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SLIP flight dynamics
function xdot = dynamics_f(~, x_cart, params)
    
    % cartesian state, x = [x, z, x_dot, z_dot]
    x_dot = x_cart(3);
    z_dot = x_cart(4);

    % drift dynamics
    xdot = [x_dot;
            z_dot;
            0;
            -params.g];
end

% SLIP ground dynamics
function xdot = dynamics_g(t, x_polar, params)
    
    % unpack the system parameters
    m = params.m;
    g = params.g;
    l0 = params.l0;

    % get the spring stiffness
    k = spring_stiffness(t, x_polar, params);

    % polar state, x = [r, theta, r_dot, theta_dot]
    r = x_polar(1);
    theta = x_polar(2);
    r_dot = x_polar(3);
    theta_dot = x_polar(4);

    xdot = [r_dot;
            theta_dot;
            r*theta_dot^2 - g*cos(theta) + (k/m)*(l0 - r);
            -(2/r)*r_dot*theta_dot + (g/r)*sin(theta)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONTROL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% spring stiffness input
function k = spring_stiffness(~, x_polar, params)
    k = 15000;
end

% angle control input
function alpha = angle_control(t_abs, x_cart, params)
    
    % unpack the desired state
    x_des = trajectory(t_abs, params);
    px_des = x_des(1);
    vx_des = x_des(2);

    % simple Raibert controller
    K = params.K;
    K_p = K(1);
    K_v = K(2);

    % compute the desired angle
    px_actual = x_cart(1);
    vx_actual = x_cart(3);
    alpha = K_p * (px_actual - px_des) + K_v * (vx_actual - vx_des);

    % clip the angle to a range
    alpha_low = -params.alpha_max;
    alpha_high = params.alpha_max;
    alpha = max(alpha_low, min(alpha_high, alpha));
end

% get the desired state based on some desired trajectory based on absolute time
function x_traj = trajectory(t_abs, params)

    % desired position and velocity
    if params.traj_type == "pos"
        px_des = params.p_des;
        vx_des = 0;
    elseif params.traj_type == "vel"
        vx_des = params.v_des;
        px_des = vx_des * t_abs; % TODO: only works for constant velocity, generalize
    end

    % desired state
    x_traj = [px_des; vx_des];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GUARDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% guard: flight to ground
function [value, isterminal, direction] = flight_to_ground(t_abs, x_cart, alpha, params)

    % to determine if the SLIP foot has hit the ground
    z_com = x_cart(2);
    foot_height = z_com - params.l0 * cos(alpha);

    % guard conditions
    value = foot_height;  % foot height at ground
    isterminal = 1;       % 1: stop integrating
    direction = -1;       % direction 
end

% guard: ground to flight, leg condition
function [value, isterminal, direction] = ground_to_flight(~, x_polar, params)

    % equivalent representation in cartesian coordinates
    x_cart = polar_to_cart(x_polar);

    % leg length is uncompressed, r = l0
    l = [x_cart(1); x_cart(2)];
    leg_length = norm(l);                        % Euclidean length of the leg
    compressed_length = leg_length - params.l0;  % difference from nominal uncompressed length

    % taking off condition, vel >= 0
    xdot = x_cart(3); 
    zdot = x_cart(4);  
    v_com = [xdot; zdot];  
    vel = l' * v_com;        % velocity of the CoM along the leg direction

    if compressed_length >= 0
        if vel >= 0
            value = 0;
        else
            value = 1;
        end
    else
        value = 1;
    end

    % Ensure the solver stops when both conditions are met
    isterminal = 1;  % Stop integration when event is triggered
    direction =  0;  % compressed_length must be increasing (zero-crossing from positive to negative)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Poincare section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% apex: detect the peak of the flight phase
function [value, isterminal, direction] = poincare_section(~, x_cart)

    % poincare section
    z_dot = x_cart(4);

    value = z_dot;  % z_dot = 0
    isterminal = 0; % stop integrating
    direction = -1; % negative direction

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% convert caterisan <---> polar coordinates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert a cartesian state to polar state where the origin is at the foot
function x_polar = cart_to_polar(x_cart, params, alpha)
    
    % flight state, x = [x, z, x_dot, z_dot]
    x = x_cart(1);
    z = x_cart(2);
    xdot = x_cart(3);
    zdot = x_cart(4);

    % positions
    p_com = [x; z];  % CoM position
    p_foot = [x + params.l0 * sin(alpha); z - params.l0 * cos(alpha)]; % foot position

    x = p_com(1) - p_foot(1);
    z = p_com(2) - p_foot(2);

    r = sqrt(x^2 + z^2);
    th = atan2(x, z);     % be carefule about arctan2
    rdot = (x*xdot + z*zdot) / r;
    thdot = (xdot*z - x*zdot) / r^2;

    x_polar = [r; th; rdot; thdot];
end

% convert a polar state to cartesian state, where the origin is at the foot
function x_cart = polar_to_cart(x_polar)
    
    % ground state, x = [r, theta, r_dot, theta_dot]
    r = x_polar(1);
    th = x_polar(2);
    rdot = x_polar(3);
    thdot = x_polar(4);

    x = r * sin(th);
    z =  r * cos(th);
    xdot = (rdot * sin(th) + r * thdot * cos(th));
    zdot =  rdot * cos(th) - r * thdot * sin(th);

    x_cart = [x; z; xdot; zdot];
end
