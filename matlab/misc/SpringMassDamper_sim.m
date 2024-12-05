%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simple spring-mass-damper simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;

% system parameters
params.m = 1;  % mass
params.k = 50;  % spring constant
params.b = 5;  % damping constant
params.g = 9.81;  % gravity
params.l0 = 0.5;  % nominal length

% simulation parameters
params.dt = 0.01;

% simulation parameters
t_max = 3.0;
tspan = 0:params.dt:t_max;

% initial conditions
x0 = [0.5; % inital position 
      0.0]; % initial velocity

% sample a control trajectory
mu = params.l0;
sigma = 0.05;
u_rand = normrnd(mu, sigma, length(tspan), 1);
u_t = [tspan', u_rand];

% simulate the system
[t, x] = ode45(@(t, x) dynamics(t, x, u_t, params), tspan, x0);

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
plot(u_t(:,1),u_t(:,2), 'LineWidth', 2);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% solution to a linear system 
function xdot = dynamics(t, x, u_t, params)

    % system parameters
    k = params.k;
    m = params.m;
    b = params.b;
    g = params.g;

    % define the linear system matrices
    A = [0, 1; 
         -k/m, -b/m];
    B = [0; 
         1/m];
    C = [0; 
         (k/m)*params.l0-g];

    % get the control action
    u = get_control(t, x, u_t, params);

    % define the dynamics
    xdot = A*x + B*u + C;

end

% get the piecewise constant control action
function u = get_control(t, x, u_t, params)

    % % extract the time and control values
    % t_vals = u_t(:, 1);
    % u_vals = u_t(:, 2);

    % % find which intervalm we are currently in
    % ind = find(t_vals <= t, 1, 'last');
    
    % % in an interval
    % if ind+1 <= length(t_vals)
    %     u0 = u_vals(ind);
    %     uf = u_vals(ind+1);
    %     t0 = t_vals(ind);
    %     tf = t_vals(ind+1);
    %     u = u0 + (uf-u0)/(tf-t0)*(t-t0);
    % % beyond the last interval
    % else
    %     u = u_vals(end);
    % end

    p_des = 0.2;
    v_des = 0.0;
    

    u_ff = params.m * (-(params.k/params.m) * params.l0 + params.g);
    kp = 0.0;
    kd = 0.0;
    % u = kp*(p_des - x(1)) + kd*(v_des - x(2)) + u_ff;
    u = 0;

end

