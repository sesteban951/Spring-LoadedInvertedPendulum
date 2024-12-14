%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Double Support system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% system parameters
params.m = 1.0;  % mass 
params.g = 9.81; % gravity
params.k = 150;  % spring constant
params.l0 = 1.0; % free length of the leg
params.b = 15.0;  % damping coefficient
params.p1 = [0; 0]; % left leg position
params.p2 = [2; 0]; % right leg position

% intial conditions
x0 = [1;   % px
      2;   % pz
      0;   % vx
      0];  % vz

% time span
rt = 1.0;         % real time rate multiplier
f = 60;            % frequency, [Hz]
dt = 1/(f);        % time step, [s]
tmax = 7.0;        % max time, [s]
tspan = 0:dt:tmax; % time span, [s]

% solve the dynamics
options = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);
[t, x] = ode45(@(t, x) dynamics(t, x, params), tspan, x0);

% save the data into a csv
csvwrite('data/newton.csv', [t, x]);

% plot the data
figure('Name', 'Animation', 'Position', [100, 100, 1200, 800]);
set(0, 'DefaultFigureRenderer', 'painters');

subplot(2, 2, 1);
plot(t, x(:, 1), 'b', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('t'); ylabel('px');
grid on; axis equal;

subplot(2, 2, 3);
plot(t, x(:, 2), 'b', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('t'); ylabel('pz');
grid on; axis equal;

subplot(2, 2, [2,4]);
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
xlim([px_min-1, px_max+1]);
ylim([pz_min-1, pz_max+1]);


tic;
ind = 1;
t = t * (1/rt);
while true

    % draw the legs
    leg1 = plot([params.p1(1), x(ind, 1)], [params.p1(2), x(ind, 2)], 'k', 'LineWidth', 2);
    leg2 = plot([params.p2(1), x(ind, 1)], [params.p2(2), x(ind, 2)], 'k', 'LineWidth', 2);

    % draw the ball mass
    mass = plot(x(ind, 1), x(ind, 2), 'ko', 'MarkerSize', 20, 'MarkerFaceColor', [0.8500 0.3250 0.0980]	);

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
    u1 = -50;
    u2 = 50;

    % compute the dynamics
    a_com = r1_hat * ((k/m) * (l0 - r1_norm) - (b/m) * (v_com' * r1) / r1_norm + (1/m) * u1) ...
          + r2_hat * ((k/m) * (l0 - r2_norm) - (b/m) * (v_com' * r2) / r2_norm + (1/m) * u2) ...
          + [0; -g];
    xdot = [v_com; a_com];

end


