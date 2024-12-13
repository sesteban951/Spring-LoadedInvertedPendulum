%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Double Support system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% system parameters
params.m = 1.0;  % mass 
params.g = 9.81; % gravity
params.k = 10;  % spring constant
params.l0 = 0.5; % free length of the leg

% intial conditions
x0 = [0.5;   % r
      pi/2;     % theta
      0;     % rdot
      0.0];  % thetadot
d = 1.0;

% time span
f = 40;            % frequency, [Hz]
dt = 1/(f);        % time step, [s]
tmax = 5.0;        % max time, [s]
tspan = 0:dt:tmax; % time span, [s]

[dgdr1, dgdt1] = make_lagrangian_gradients(params);

val1 = [dgdr1(x0(1), x0(2), d), dgdt1(x0(1), x0(2), d)];
val2 = [eval_dgdr(x0, d, params), eval_dgdt(x0, d, params)];

% solve the dynamics
[t, x] = ode45(@(t, x) dynamics(t, x, d, params, dgdr1, dgdt1), tspan, x0);

% convert to cartesian
for i = 1:length(t)
    x(i, :) = polart_to_cartesian(x(i, :));
end

% plot the data
figure('Name', 'Animation');
set(0, 'DefaultFigureRenderer', 'painters');

subplot(2, 2, 1);
quiver(x(1:end-1, 1), x(1:end-1, 2), diff(x(:, 1)), diff(x(:, 2)), 'r', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('px'); ylabel('py');
grid on; axis equal;

subplot(2, 2, 3);
quiver(x(1:end-1, 3), x(1:end-1, 4), diff(x(:, 3)), diff(x(:, 4)), 'r', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('vx'); ylabel('vy');
grid on; axis equal;

subplot(2, 2, [2,4]);
hold on;
plot([-0.1, 0.1], [0, 0], 'k', 'LineWidth', 2);
xlabel('px'); ylabel('py');
grid on; axis equal;
xlim([-2, 2]);
ylim([-2, 2]);

tic;
ind = 1;
while true

    % draw the pole
    pole = plot([0, x(ind, 1)], [0, x(ind, 2)], 'k', 'LineWidth', 2);

    % draw the ball mass
    mass = plot(x(ind, 1), x(ind, 2), 'ko', 'MarkerSize', 20, 'MarkerFaceColor', [0.8500 0.3250 0.0980]	);

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
        delete(mass);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% double support dynamics
function xdot = dynamics(t, x, d, params, dgdr, dgdt)

    % unpack the parameters
    m = params.m;
    g = params.g;
    k = params.k;
    l0 = params.l0;

    % unpack the state
    r = x(1);
    theta = x(2);
    rdot = x(3);
    thetadot = x(4);

    % get lagrangian gradients
    dgdr_ = eval_dgdr(x, d, params);
    dgdt_ = eval_dgdt(x, d, params);
    % dgdr_ = dgdr(r, theta, d);
    % dgdt_ = dgdt(r, theta, d);

    % compute the accelerations
    xdot = [rdot; 
            thetadot; 
            r * thetadot^2 - g * cos(theta) + (k/m) * (l0 - r) + (1/m) * dgdr_; 
            -(2/r)*rdot*thetadot + (g/r)*sin(theta) + (1/(m*r^2))* dgdt_];
end

% convert the polar coordinates to cartesian
function x_cart = polart_to_cartesian(x_polar)
    
    % unpack the state
    r = x_polar(1);
    theta = x_polar(2);
    rdot = x_polar(3);
    thetadot = x_polar(4);

    % convert to cartesian
    x_cart = [r*sin(theta); 
              r*cos(theta); 
              rdot*sin(theta) + r*thetadot*cos(theta); 
              rdot*cos(theta) - r*thetadot*sin(theta)]';
    
end

% make lagrangian gradients
function [dgdr, dgdt] = make_lagrangian_gradients(params)

    % unpack the parameters
    k = params.k;
    l0 = params.l0;

    syms r theta d
    term = atan((r*sin(theta) -d)/ (r*cos(theta)));
    g = -(1/2) * k * (l0 - (r*cos(theta))/(cos(term)));

    dgdr_ = diff(g, r);
    dgdt_ = diff(g, theta);

    dgdr = matlabFunction(dgdr_, 'vars', {'r', 'theta', 'd'});
    dgdt = matlabFunction(dgdt_, 'vars', {'r', 'theta', 'd'});
end

function dgdr = eval_dgdr(x, d, params)
    % Compute the value of the mathematical expression:
    %
    % -\frac{k (sin(theta) d - r)}{2 cos(theta) r \sqrt{(d^2 - 2 sin(theta) d r + r^2) / (r^2 (cos(theta))^2)}}
    %

    % define the variables
    k = params.k;

    % extract the state
    r = x(1);
    theta = x(2);

    numerator = k * (sin(theta) * d - r);
    denominator = 2 * cos(theta) * r * sqrt((d^2 - 2 * sin(theta) * d * r + r^2) / (r^2 * (cos(theta))^2));
    
    dgdr = -numerator / denominator;
end

% function dgdt = eval_dgdt(k, theta, d, r)
function dgdt = eval_dgdt(x, d, params)
    % Compute the value of the mathematical expression:
    %
    % -\frac{kd}{2\sqrt{(d^2 - 2 sin(theta) dr + r^2) / (r^2 (cos(theta))^2)}}
    %

    % define the variables
    k = params.k;

    % extract the state
    r = x(1);
    theta = x(2);

    numerator = k * d;
    denominator = 2 * sqrt((d^2 - 2 * sin(theta) * d * r + r^2) / (r^2 * (cos(theta))^2));
    
    dgdt = -numerator / denominator;
end
