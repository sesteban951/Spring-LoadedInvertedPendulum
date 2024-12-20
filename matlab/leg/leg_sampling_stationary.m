%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample the leg trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc; 

% plotting parameters
tot_time = 3.0; % real time rate

% bezier curve parameters
deg = 15;               % polynomial degree
bounds = [0.8, 1.2;     % bounds
          -pi/4, pi/4;  
          -pi/4, pi/4];
          % TODO: Try gaussian sampling

[T, B, P] = generate_trajectory(deg, bounds);
[T, p_foot] = get_foot_positions(T, B);
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
plot3([0, 0], [0, 0], [0, 0], 'ko', 'MarkerSize', 20, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);

% plot the foot positions
ind = 1;
tic;
while ind <= length(T)
    
    % plot the foot positions
    p = p_foot(:, ind);
    foot_trail = plot3(p(1), p(2), p(3), 'm.', 'MarkerSize', 5);
    foot = plot3(p(1), p(2), p(3), 'k.', 'MarkerSize', 20);
    pole = plot3([0, p(1)], [0, p(2)], [0, p(3)], 'k-', 'LineWidth', 2);
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
        delete(foot);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute the binomial coefficients of bezier curve
function c = bezier_coeff(n , k)
    c = factorial(n) / (factorial(k) * factorial(n - k));
end

% genearte random trajectory
% https://en.wikipedia.org/wiki/B%C3%A9zier_curve
function [T, B, P] = generate_trajectory(deg, bounds)
    
    % degree of the bezier curve
    dim = size(bounds, 1);

    % generate random control points
    T = linspace(0, 1, 150);
    B = zeros(dim, length(T));

    % generate random control points
    P = zeros(dim, deg + 1);
    for i = 1:dim
        lb = bounds(i, 1);
        ub = bounds(i, 2);
        P(i, :) = unifrnd(lb, ub, 1, deg + 1);
    end

    % fix the end points
    % P(:, 1) = [1;0;0];
    % P(:, end) = [1;0;0];

    % compute the bezier curve
    for i = 0:deg
        Pi = P(:, i + 1);
        ci = bezier_coeff(deg, i);
        term = ci * Pi * (1 - T).^(deg - i) .* T.^i;
        B = B + term;
    end
end