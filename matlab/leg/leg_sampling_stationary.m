%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample the leg trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc; 

% plotting parameters
tot_time = 3.0; % real time rate

% bezier curve parameters
deg = 10;               % polynomial degree
bounds = [0.8, 1.2;     % bounds
          -pi/4, pi/4;  
          -pi/4, pi/4];
          % TODO: Try gaussian sampling

n_knots = 250;
[T, P, B, Bdot] = bezier_curve(deg, bounds, n_knots);
[T, p_foot] = get_foot_positions(T, B);
[T, v_foot] = get_foot_velocities(T, B, Bdot);
T = T * tot_time;

% plot the leg positions
figure('WindowState', 'maximized');
set(gcf,'renderer','painters')

% plot the radius
subplot(3, 2, 1);
plot(T, B(1, :), 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Radius (m)');
title('Radius');
grid on;

% plot the angles
subplot(3, 2, 3);
plot(T, B(2, :), 'b', 'LineWidth', 2);
xlabel('Time (s)'); 
ylabel('Roll (rad)');
title('Roll Angle');
grid on;

subplot(3, 2, 5);
plot(T, B(3, :), 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Pitch (rad)');
title('Pitch Angle');
grid on;

% plot animation
subplot(3, 2, [2, 4, 6]);
hold on; axis equal; grid on;
view(0, 30);

% plot the x, y, z axes
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);
xlabel('x'); ylabel('y'); zlabel('z');

% plot S2 sphere
% [X_sphere, Y_sphere, Z_sphere] = sphere;
% surf(X_sphere, Y_sphere, Z_sphere, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.1);

% plot a ball at the origin
plot3([0, 0], [0, 0], [0, 0], 'ko', 'MarkerSize', 30, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);

% plot the foot positions
ind = 1;
tic;
while ind <= length(T)
    
    % plot the foot positions
    p = p_foot(:, ind);
    foot_trail = plot3(p(1), p(2), p(3), 'm.', 'MarkerSize', 5);
    foot_pos = plot3(p(1), p(2), p(3), 'k.', 'MarkerSize', 20);
    pole = plot3([0, p(1)], [0, p(2)], [0, p(3)], 'k-', 'LineWidth', 2);
    
    % plot the foot velocities as quiver
    v = v_foot(:, ind);
    v = 0.5 * v / norm(v);
    foot_vel = quiver3(p(1), p(2), p(3), v(1), v(2), v(3), 'c', 'LineWidth', 2);
    
    drawnow;

    % change the view
    view((T(ind) / T(end)) * 135, 30);

    % set the title
    msg = sprintf('Time: %.2f s', T(ind));
    title(msg);

    % wait until the next frame
    while toc < T(ind+1)
        % wait
    end

    % increment the index
    if ind+1 >= length(T)
        break;
    else
        ind = ind + 1;
        delete(foot_pos);
        delete(foot_vel);
        delete(pole);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get foor positions given bezier curve
function [T, p_foot] = get_foot_positions(T, B)
    
    % length of the trajectory
    n = length(T);

    % compute the foot positions
    p_foot = zeros(3, n);
    for i = 1:n
        % configuration vector
        q = B(:, i);
        
        % extract the joint angles
        r = q(1);
        phi = q(2); % roll
        psi = q(3); % pitch

        % zero position of the leg
        p_zero = [0; 0; -r];

        % compute the rotation matrices
        R_phi = [1, 0, 0;
                 0, cos(phi), -sin(phi);
                 0, sin(phi), cos(phi)];
        R_psi = [cos(psi),  0, sin(psi);
                 0,         1, 0;
                 -sin(psi), 0, cos(psi)];
        
        % final position of the leg
        p_final = R_psi * R_phi * p_zero;

        % store the foot position
        p_foot(:, i) = p_final;
    end
end

% get foot velocities given bezier curve
function [T, v_foot] = get_foot_velocities(T, B, Bdot)

    % comput the trig functions
    sin_phi = sin(B(2, :));
    sin_psi = sin(B(3, :));
    cos_phi = cos(B(2, :));
    cos_psi = cos(B(3, :));
    
    % get state
    r = B(1, :);
    rdot = Bdot(1, :);
    phidot = Bdot(2, :);
    psidot = Bdot(3, :);

    % compute the foot velocities
    vx_foot = -(rdot .* cos_phi .* sin_psi) + (r .* phidot .* sin_phi .* sin_psi) - (r .* psidot .* cos_phi .* cos_psi);
    vy_foot = rdot .* sin_phi + r .* phidot .* cos_phi;
    vz_foot = -(rdot .* cos_phi .* cos_psi) + (r .* phidot .* sin_phi .* cos_psi) + (r .* psidot .* cos_phi .* sin_psi);

    v_foot = [vx_foot; vy_foot; vz_foot];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute the binomial coefficients of bezier curve
function c = bezier_coeff(n , k)
    c = factorial(n) / (factorial(k) * factorial(n - k));
end

% generate random trajectory
% https://en.wikipedia.org/wiki/B%C3%A9zier_curve
% https://pomax.github.io/bezierinfo/#derivatives
function [T, P, B, Bdot] = bezier_curve(deg, bounds, n_knots)
    
    % degree of the bezier curve
    dim = size(bounds, 1);
    n = deg;

    % generate random control points
    T = linspace(0, 1, n_knots);
    
    % generate random control points
    P = zeros(dim, n + 1);
    for i = 1:dim
        lb = bounds(i, 1);
        ub = bounds(i, 2);
        P(i, :) = unifrnd(lb, ub, 1, n + 1);
    end
    
    % compute the bezier curve positions
    B = zeros(dim, length(T));
    for i = 0:n
        ci = bezier_coeff(n, i);
        Pi = P(:, i + 1);
        term = ci * Pi * (1 - T).^(n - i) .* T.^i;
        B = B + term;
    end

    % compute the bezier curve velocities
    k = n - 1;
    Bdot = zeros(dim, length(T));
    for i = 0:k
        ci = bezier_coeff(k, i);
        Pi = P(:, i+2) - P(:, i+1);
        term = n * ci * Pi * (1 - T).^(k - i) .* T.^i;
        Bdot = Bdot + term;
    end
end