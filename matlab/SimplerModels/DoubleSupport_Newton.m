%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Double Support system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% system parameters
params.m = 25.0;   % mass 
params.g = 9.81;   % gravity
params.l0 = 1.5;   % free length of the leg
params.k = 5000;  % spring constant
params.b = 10.0;    % damping coefficient
params.p1 = [0; 0]; % left leg position
params.p2 = [0.5; 0]; % right leg position

% intial conditions
x0 = [0.25; % px
      1.5;  % pz
      1;    % vx
      10];  % vz

% time span
rt = 1.0;         % real time rate multiplier
f = 25;            % frequency, [Hz]
dt = 1/(f);        % time step, [s]
tmax = 3.0;        % max time,  [s]
tspan = 0:dt:tmax; % time span, [s]

% Dummy solve to warm up the ode45 solver
[t, x] = ode45(@(t, x) dynamics(t, x, params), tspan, x0);

% ODE45 solver
% options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
tic;
[t, x] = ode45(@(t, x) dynamics(t, x, params), tspan, x0);
msg_ODE45 = sprintf('ode45 time: %.3f [ms]', toc * 1000);

% RK4 solver
tic;
[t_RK4, x_RK4] = RK4(tspan, x0, params);
msg_RK4 = sprintf('RK4 time: %.3f [ms]', toc * 1000);

% RK3 solver
tic;
[t_RK3, x_RK3] = RK3(tspan, x0, params);
msg_RK3 = sprintf('RK3 time: %.3f [ms]', toc * 1000);

% RK2 solver
tic;
[t_RK2, x_RK2] = RK2(tspan, x0, params);
msg_RK2 = sprintf('RK2 time: %.3f [ms]', toc * 1000);

% Euler solver
tic;
[t_Euler, x_Euler] = euler(tspan, x0, params);
t_Euler = t_Euler * 0;
x_Euler = x_Euler * 0;
msg_Euler = sprintf('Euler time: %.3f [ms]', toc * 1000);

% display the time
disp(msg_Euler);
disp(msg_RK2);
disp(msg_RK3);
disp(msg_RK4);
disp(msg_ODE45);

% save the data into a csv
csvwrite('data/newton.csv', [t, x]);

% plot the data
figure('Name', 'Animation', 'WindowState', 'maximized');
set(0, 'DefaultFigureRenderer', 'painters');

subplot(2, 3, 1);
hold on;
plot(t, x(:, 1), 'b', 'LineWidth', 1);
plot(t_RK2, x_RK2(:, 1), 'r', 'LineWidth', 1);
plot(t_RK3, x_RK3(:, 1), 'm', 'LineWidth', 1); 
plot(t_RK4, x_RK4(:, 1), 'g', 'LineWidth', 1);
plot(t_Euler, x_Euler(:, 1), 'm', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('t'); ylabel('px');
legend('ode45', 'RK2', 'RK3', 'RK4', 'Euler');
grid on;

subplot(2, 3, 2);
hold on;
plot(t, x(:, 2), 'b', 'LineWidth', 1);
plot(t_RK2, x_RK2(:, 2), 'r', 'LineWidth', 1);
plot(t_RK3, x_RK3(:, 2), 'm', 'LineWidth', 1);
plot(t_RK4, x_RK4(:, 2), 'g', 'LineWidth', 1);
plot(t_Euler, x_Euler(:, 2), 'm', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('t'); ylabel('pz');
legend('ode45', 'RK2', 'RK3', 'RK4', 'Euler');
grid on;

subplot(2, 3, 4);
hold on;
plot(t, x(:, 3), 'b', 'LineWidth', 1);
plot(t_RK2, x_RK2(:, 3), 'r', 'LineWidth', 1);
plot(t_RK3, x_RK3(:, 3), 'm', 'LineWidth', 1);
plot(t_RK4, x_RK4(:, 3), 'g', 'LineWidth', 1);
plot(t_Euler, x_Euler(:, 3), 'm', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('t'); ylabel('vx');
legend('ode45', 'RK2', 'RK3', 'RK4', 'Euler');
grid on;

subplot(2, 3, 5);
hold on;
plot(t, x(:, 4), 'b', 'LineWidth', 1);
plot(t_RK2, x_RK2(:, 4), 'r', 'LineWidth', 1);
plot(t_RK3, x_RK3(:, 4), 'm', 'LineWidth', 1);
plot(t_RK4, x_RK4(:, 4), 'g', 'LineWidth', 1);
plot(t_Euler, x_Euler(:, 4), 'm', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('t'); ylabel('vz');
legend('ode45', 'RK2', 'RK3', 'RK4', 'Euler');
grid on;

subplot(2, 3, [3,6]);
hold on;
plot(params.p1(1), params.p1(2), 'kx', 'MarkerSize', 10);
plot(params.p2(1), params.p2(2), 'kx', 'MarkerSize', 10);
xline(0, '--');
yline(0, '--');
xlabel('px'); ylabel('py');

px_min = min(x(:, 1));
px_max = max(x(:, 1));
pz_min = min(x(:, 2));
pz_max = max(x(:, 2));
grid on; axis equal;
xlim([px_min-1.5, px_max+1.5]);
ylim([pz_min-1.5, pz_max+1.5]);

tic;
ind = 1;
t = t * (1/rt);
while true

    px = x(ind, 1);
    pz = x(ind, 2);

    % draw the legs
    leg1 = plot([params.p1(1), px], [params.p1(2), pz], 'k', 'LineWidth', 2);
    leg2 = plot([params.p2(1), px], [params.p2(2), pz], 'k', 'LineWidth', 2);

    % draw the ball mass
    mass = plot(px, pz, 'ko', 'MarkerSize', 20, 'MarkerFaceColor', [0.8500 0.3250 0.0980]	);

    drawnow;

    % put the time in the title
    msg = sprintf('Time: %.2f sec', t(ind) * rt);
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
        delete(leg1);
        delete(leg2);
        delete(mass);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% double support dynamics
function xdot = dynamics(t, x, params)

    % unpack the parameters
    m = params.m;
    g = params.g;
    k = params.k;
    b = params.b;
    l0 = params.l0;
    p1 = params.p1;
    p2 = params.p2;

    % unpack state
    p_com = [x(1); x(2)];
    v_com = [x(3); x(4)];

    % compute the leg vectors
    r1 = p_com - p1;
    r2 = p_com - p2;
    r1_norm = norm(r1);
    r2_norm = norm(r2);
    r1_hat = r1/r1_norm;
    r2_hat = r2/r2_norm;
    
    % compute control effort
    u1 = 10;
    u2 = 10;

    % compute the dynamics
    a_com = r1_hat * ((k/m) * (l0 - r1_norm) - (b/m) * (v_com' * r1) / r1_norm + (1/m) * u1) ...
          + r2_hat * ((k/m) * (l0 - r2_norm) - (b/m) * (v_com' * r2) / r2_norm + (1/m) * u2) ...
          + [0; -g];
    xdot = [v_com; a_com];

end

% custom euler integrator
function [t, x] = euler(tspan, x0, params)

    % make the containers
    t = tspan;
    x = zeros(length(t), length(x0));

    % initial conditions
    x(1, :) = x0;
    xk = x0;

    % take euler steps
    for k = 1:length(t)-1

        % get the dt
        dt = t(k+1) - t(k);

        % euler integration
        xk = xk + dt*dynamics(t(k), x(k, :), params);

        % store the results
        x(k+1, :) = xk;
    end
end

% custom RK2 integrator
function [t, x] = RK2(tspan, x0, params)

    % make the containers
    t = tspan;
    x = zeros(length(t), length(x0));

    % initial conditions
    x(1, :) = x0;
    xk = x0;

    % take RK2 steps
    for k = 1:length(t)-1

        % get the dt
        dt = t(k+1) - t(k);

        % RK2 integration
        f1 = dynamics(t(k), xk, params);
        f2 = dynamics(t(k) + 0.5*dt, xk + 0.5*dt*f1 ,params);
        xk = xk + dt*f2;

        % store the results
        x(k+1, :) = xk;
    end
end

% custom RK3 integrator
function [t, x] = RK3(tspan, x0, params)

    % make the containers
    t = tspan;
    x = zeros(length(t), length(x0));

    % initial conditions
    x(1, :) = x0;
    xk = x0;

    % take RK3 steps
    for k = 1:length(t)-1

        % get the dt
        dt = t(k+1) - t(k);

        % RK3 integration
        f1 = dynamics(t(k), xk, params);
        f2 = dynamics(t(k) + 0.5*dt, xk + 0.5*dt*f1, params);
        f3 = dynamics(t(k) + dt, xk - dt*f1 + 2*dt*f2, params);
        xk = xk + (dt/6)*(f1 + 4*f2 + f3);

        % store the results
        x(k+1, :) = xk;
    end
end
 

% custom RK4 integrator
function [t, x] = RK4(tspan, x0, params)

    % make the containers
    t = tspan;
    x = zeros(length(t), length(x0));

    % initial conditions
    x(1, :) = x0;
    xk = x0;

    % take RK4 steps
    for k = 1:length(t)-1

        % get the dt
        dt = t(k+1) - t(k);

        % RK4 integration
        f1 = dynamics(t(k), xk, params);
        f2 = dynamics(t(k) + 0.5*dt, xk + 0.5*dt*f1 ,params);
        f3 = dynamics(t(k) + 0.5*dt, xk + 0.5*dt*f2 ,params);
        f4 = dynamics(t(k) + dt, xk + dt*f3 ,params);
        xk = xk + (dt/6)*(f1 + 2*f2 + 2*f3 + f4);

        % store the results
        x(k+1, :) = xk;
    end
end