%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simple spring-mass-damper simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;

% system parameters
sys.m = 35;     % mass
sys.k = 5000;   % spring constant
sys.b = 500;    % damping constant
sys.g = 9.81;   % gravity
sys.l0 = 0.65;  % nominal length

% mpc parameters
mpc.K = 100;      % number of rollouts
mpc.N = 200;      % number of time steps
mpc.dt = 0.01;    % time step
mpc.interp = 'L'; % interpolation type

% simulation parameters
tspan = 0:mpc.dt:mpc.dt*(mpc.N-1);
rt = 1.0; % real time rate

% initial conditions
x0 = [0.75; % inital position 
      0.0]; % initial velocity

% simulate the system 
[t, x, u_t] = rollout(x0, sys, mpc);

animate(t, x, u_t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROLLOUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% perform a rollout
function [t, x_t, u_t] = rollout(x0, sys, mpc)

    % generate a random input signal
    u_t = sample_input(mpc);

    % generate the tpsan
    tspan = 0:mpc.dt:mpc.dt*(mpc.N-1);

    % simulate the system
    [t, x_t] = ode45(@(t, x) dynamics(t, x, u_t, sys, mpc), tspan, x0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMICS AND CONTROL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dynamics of the linear system
function xdot = dynamics(t, x, U, sys, mpc)

    % system parameters
    k = sys.k;
    m = sys.m;
    b = sys.b;
    g = sys.g;

    % define the linear system matrices
    A = [0, 1; 
         -k/m, -b/m];
    B = [0; 
         1/m];
    C = [0; 
         (k/m)*sys.l0-g];

    % get the control action
    v = interpolate_control(t, U, mpc);
    u = k * (v - sys.l0);

    % define the dynamics
    xdot = A*x + B*u + C;
end

% sample a control signal
function U = sample_input(mpc)

    % sample a random control signal
    lb = 0.5;
    ub = 0.7;
    U = unifrnd(lb, ub, 1, mpc.N-1);
end

% get the piecewise constant control action
function u = interpolate_control(t, U, mpc)

    % extract the time and control values
    t_vals = 0:mpc.dt:mpc.dt*(mpc.N-2);
    u_vals = U;

    % find which intervalm we are currently in
    ind = find(t_vals <= t, 1, 'last');
    
    % linear interpolation
    if mpc.interp == 'L'
        if ind+1 <= length(t_vals) % t inside an interval
            u0 = u_vals(ind);
            uf = u_vals(ind+1);
            t0 = t_vals(ind);
            tf = t_vals(ind+1);
            u = u0 + (uf-u0)/(tf-t0)*(t-t0);
        else                       % t beyond last interval
            u = u_vals(end);
        end
    % zero-order hold interpolation
    elseif mpc.interp == 'Z'
        u = u_vals(ind);
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANIMATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function animate(t, x, u_t)

    % plot the results
    figure('Name', 'Spring-Mass-Damper Simulation');

    subplot(3, 2, 1);
    plot(t, x(:, 1), 'LineWidth', 2);
    xlabel('Time [s]');
    ylabel('Position [m]');
    grid on;

    subplot(3, 2, 3);
    plot(t, x(:, 2), 'LineWidth', 2);
    xlabel('Time [s]');
    ylabel('Velocity [m/s]');
    grid on;

    subplot(3, 2, 5);
    plot(t(1:end-1), u_t, 'LineWidth', 2);
    xlabel('Time [s]');
    ylabel('Control [N]');
    grid on;

    subplot(3, 2, [2,4,6]);
    hold on;
    yline(0); 
    yline(0.75, '--');
    yline(-0.75, '--');
    ylabel('Position [m]');
    axis equal; grid on;

    tic;
    ind = 1;
    while 1==1

        % draw the mass spring damper system
        pole = plot([0, 0], [0, x(ind, 1)], 'k', 'LineWidth', 2);
        
        box_center = [0; x(ind, 1)];
        box = rectangle('Position', [box_center(1)-0.1, box_center(2)-0.1, 0.2, 0.2], 'Curvature', 0.1, 'FaceColor', 'r');
        
        drawnow;

        % put the time in the title
        msg = sprintf('Time: %.2f sec', t(ind));
        title(msg);

        % wait until the next time step
        while toc < t(ind+1)
            % end
        end

        % increment the index
        if ind+1 >= length(t)
            break;
        else
            ind = ind + 1;
            delete(pole);
            delete(box);
        end
    end
end
