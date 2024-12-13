%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single Support system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% system parameters
params.m = 1.0;  % mass 
params.g = 9.81; % gravity
params.k = 100;  % spring constant
params.l0 = 1.0; % free length of the leg

% intial conditions
x0 = [1;     % r
      pi/2;  % theta
      0;     % rdot
      1.0];    % thetadot

% time span
f = 40;            % frequency, [Hz]
dt = 1/(f);     % time step, [s]
tmax = 10;         % max time, [s]
tspan = 0:dt:tmax; % time span, [s]

% solve the dynamics
[t, x] = ode45(@(t, x) dynamics(t, x, params), tspan, x0);

% convert to cartesian
for i = 1:length(t)
    x(i, :) = polart_to_cartesian(x(i, :));
end

% plot the data
figure('Name', 'Animation');
set(0, 'DefaultFigureRenderer', 'painters');

subplot(2, 2, 1);
quiver(x(1:end-1, 1), x(1:end-1, 2), diff(x(:, 1)), diff(x(:, 2)), 'r', 'LineWidth', 1);
xlabel('px'); ylabel('py');
grid on; axis equal;

subplot(2, 2, 3);
quiver(x(1:end-1, 3), x(1:end-1, 4), diff(x(:, 3)), diff(x(:, 4)), 'r', 'LineWidth', 1);
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

% single support dynamics
function xdot = dynamics(t, x, params)

    % unpack the system parameters
    m = params.m;
    g = params.g;
    l0 = params.l0;
    k = params.k;

    % unpack the state
    r = x(1);
    theta = x(2);
    rdot = x(3);
    thetadot = x(4);

    % define the dynamics
    xdot = [rdot; 
            thetadot; 
            r * thetadot^2 + (k/m)*(l0 - r) - g*cos(theta);
            -2*rdot*thetadot/r + g*sin(theta)/r];
end

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
