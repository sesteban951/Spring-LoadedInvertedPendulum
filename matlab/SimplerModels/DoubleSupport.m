%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Double Support system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% system parameters
params.m = 1.0;  % mass 
params.g = 9.81; % gravity
params.k = 10;  % spring constant
params.l0 = 1.0; % free length of the leg
params.b = 0.5;  % damping coefficient
params.d = 2.0;  % distance between the legs
d = params.d;

% intial conditions
x0 = [1;   % r
      pi/2;  % theta
      0;     % rdot
      1.0];  % thetadot

% time span
rt = 2.00;         % real time rate multiplier
f = 40;            % frequency, [Hz]
dt = 1/(f);        % time step, [s]
tmax = 10.0;        % max time, [s]
tspan = 0:dt:tmax; % time span, [s]

[dgdr, dgdt] = make_lagrangian_gradients(params);

% val1 = [dgdr1(x0(1), x0(2)), dgdt1(x0(1), x0(2))];
% val2 = [eval_dgdr(x0, d, params), eval_dgdt(x0, d, params)];

% solve the dynamics
[t, x] = ode45(@(t, x) dynamics(t, x, params, dgdr, dgdt), tspan, x0);

% get right leg position
p_R = zeros(length(t), 2);
for i = 1:length(t)
    [px, pz] = right_leg_position(x(i, :), params);
    p_R(i, 1:2) = [px, pz];
end

% convert to cartesian
for i = 1:length(t)
    x(i, :) = polart_to_cartesian(x(i, :));
end

% plot the data
figure('Name', 'Animation', 'Position', [100, 100, 1200, 800]);
set(0, 'DefaultFigureRenderer', 'painters');

subplot(2, 2, 1);
% quiver(x(1:end-1, 1), x(1:end-1, 2), diff(x(:, 1)), diff(x(:, 2)), 'r', 'LineWidth', 1);
plot(t, x(:, 1), 'b', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('t'); ylabel('px');
grid on; axis equal;

subplot(2, 2, 3);
% quiver(x(1:end-1, 3), x(1:end-1, 4), diff(x(:, 3)), diff(x(:, 4)), 'r', 'LineWidth', 1);
plot(t, x(:, 2), 'b', 'LineWidth', 1);
xline(0, '--');
yline(0, '--');
xlabel('t'); ylabel('pz');
grid on; axis equal;

subplot(2, 2, [2,4]);
hold on;
plot([-0.1, 0.1], [0, 0], 'k', 'LineWidth', 2);
xlabel('px'); ylabel('py');
grid on; axis equal;
xlim([-4, 4]);
ylim([-4, 4]);

tic;
ind = 1;
t = t * (1/rt);
while true

    % draw the left leg
    left_leg = plot([0, x(ind, 1)], [0, x(ind, 2)], 'k', 'LineWidth', 2);

    % draw the right leg
    right_leg = plot([d, p_R(ind, 1)], [0, p_R(ind, 2)], 'b', 'LineWidth', 2);

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
        delete(left_leg);
        delete(right_leg)
        delete(mass);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% double support dynamics
function xdot = dynamics(t, x, params, dgdr, dgdt)

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
    % dgdr_ = eval_dgdr(x, params); % Made using maple (less robust)
    % dgdt_ = eval_dgdt(x, params);
    dgdr_ = dgdr(r, theta);  % made using symbolic toolbox (better)
    dgdt_ = dgdt(r, theta);

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

% compute the end effector position of the right leg
function [px, pz] = right_leg_position(x_polar, params)
    
    % extract the geometry
    rL = x_polar(1);
    thetaL = x_polar(2);
    d = params.d;

    % from holonomic constraint
    thetaR = atan((rL*sin(thetaL) - d) / (rL*cos(thetaL)));
    rR = rL * cos(thetaL) / cos(thetaR);
    
    % compute the positions
    px = d + rR * sin(thetaR);
    pz = rR * cos(thetaR);

end

% make lagrangian gradients
function [dgdr, dgdt] = make_lagrangian_gradients(params)

    % unpack the parameters
    k = params.k;
    l0 = params.l0;
    d = params.d;

    syms r theta
    term = atan2((r*sin(theta) -d) , (r*cos(theta)));
    g = -(1/2) * k * (l0 - (r*cos(theta))/(cos(term)))^2;

    dgdr_ = diff(g, r);
    dgdt_ = diff(g, theta);

    dgdr = matlabFunction(dgdr_, 'vars', {'r', 'theta'});
    dgdt = matlabFunction(dgdt_, 'vars', {'r', 'theta'});
end

% compute force along the other leg
function dgdr = eval_dgdr(x, params)

    % define the variables
    k = params.k;
    d = params.d;
    l0 = params.l0;

    % extract the state
    r = x(1);
    theta = x(2);

    dgdr = k * (sin(theta) * d - r) * (-l0 + sqrt(d^2 - 2 * sin(theta) * d * r + r^2)) * (1 / sqrt(d^2 - 2 * sin(theta) * d * r + r^2));
end

% compute torque about the other leg
function dgdt = eval_dgdt(x, params)

    % define the variables
    k = params.k;
    d = params.d;
    l0 = params.l0;

    % extract the state
    r = x(1);
    theta = x(2);

    dgdt = k * d * r * cos(theta) * (-l0 + sqrt(d^2 - 2 * sin(theta) * d * r + r^2)) * (1 / sqrt(d^2 - 2 * sin(theta) * d * r + r^2));
end
