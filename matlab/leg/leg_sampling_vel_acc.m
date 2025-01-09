%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample the leg trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc; 

% distribution parameters
distr.type = 'G';
distr.bounds = [-1.0, 1.0;     % velocity rate bounds
                -1.0, 1.0];  
distr.mean = [0.0;
              0.0];
distr.std_r = 1.5;
distr.std_theta = 1.0;

% control parameters
ctrl.N = 50;
ctrl.dt = 0.05;
ctrl.interp = 'L';

% system parameters
sys.r_nom = 0.65;
sys.r_min = 0.40;
sys.r_max = 0.80;
sys.theta_min = -pi/2;
sys.theta_max = pi/2;
sys.rdot_max = 0.5;
sys.thetadot_max = 0.5;

tot_time = ctrl.dt * (ctrl.N + 1); % real time rate
tspan = 0:0.01:tot_time;

tic;
U = sample_input(ctrl, distr, sys);
disp(['Sampling time: ', num2str(toc), ' s']);

% single integrator dynamics
x0 = [sys.r_nom;  % leg length
      0.0];  % angle
[t, x] = ode45(@(t, x) single_integrator_dynamics(t, x, U, ctrl, sys), tspan, [0.65; 0.0]);

% % double integrator dynamics
% x0 = [sys.r_nom;  % leg length
%       0.0;  % angle
%       0.0;  % leg length rate
%       0.0]; % angle rate
% [t, x] = ode45(@(t, x) double_integrator_dynamics(t, x, U, ctrl, sys), tspan, x0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% post process
r = x(:, 1);
theta_rad = x(:, 2);
theta_deg = rad2deg(theta_rad);
T = (0:ctrl.N-1) * ctrl.dt;
px = -r .* sin(theta_rad);
pz = -r .* cos(theta_rad);

figure;
subplot(3, 2, 1);
hold on;
plot(t, r, 'b', 'LineWidth', 2);
% yline(sys.r_min);
% yline(sys.r_max);
xlabel('Time (s)');
ylabel('length (m)');
yline(0.65);
grid on;

subplot(3, 2, 3);
hold on;
plot(t, theta_deg, 'r', 'LineWidth', 2);
% yline(rad2deg(sys.theta_min));
% yline(rad2deg(sys.theta_max));
xlabel('Time (s)');
ylabel('angle (deg)');
yline(0);
grid on;

subplot(3, 2, 5);
hold on;
if ctrl.interp == 'L'
    plot(T, U(1, :), 'r', 'LineWidth', 2);
    plot(T, U(2, :), 'b', 'LineWidth', 2);
elseif ctrl.interp == 'Z'
    stairs(T, U(1, :), 'r', 'LineWidth', 2);
    stairs(T, U(2, :), 'b', 'LineWidth', 2);
end
legend('r', '\theta');
yline(0);
grid on;

subplot(3, 2, [2, 4, 6]);
hold on; axis equal; grid on;
xline(0);
yline(0);
x_min = min(px);
x_max = max(px);
z_min = min(pz);
z_max = max(pz);
xlim([x_min - 0.05, x_max + 0.05]);
ylim([z_min - 0.05, max(z_max, 0) + 0.05]);
plot(0, 0, 'ko', 'MarkerSize', 25, 'MarkerFaceColor', 'k');

tic;
ind = 1;
while ind <= length(t)
    % plot the foot positions
    leg = plot([0, px(ind)], [0, pz(ind)], 'b', 'LineWidth', 2);
    foot = plot(px(ind), pz(ind), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    foot_hist = plot(px(ind), pz(ind), 'b.', 'MarkerSize', 5);
    drawnow;

    msg = sprintf('Time: %.2f s', t(ind));
    title(msg);

    while toc < t(ind)
        % do nothing
    end

    ind = ind + 1;
    if ind == length(t)
        break
    else
        delete(leg);
        delete(foot);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUXILIARY FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% simple single integrator dynamics
function xdot = single_integrator_dynamics(t, x, U, ctrl, sys)
    
    % state
    r = x(1);
    theta = x(2);
    
    % input
    u = interpolate_input(t, U, ctrl, sys);
    if r <= sys.r_min
        u(1) = max(u(1), 0);
    end
    if r >= sys.r_max
        u(1) = min(u(1), 0);
    end

    if theta <= sys.theta_min
        u(2) = max(u(2), 0);
    end
    if theta >= sys.theta_max
        u(2) = min(u(2), 0);
    end
    
    rdot = u(1);
    thetadot = u(2);
    
    % dynamics
    rdotdot = rdot;
    thetadotdot = thetadot;
    
    % state derivative
    xdot = [rdotdot; thetadotdot];
end

% simple single integrator dynamics
function xdot = double_integrator_dynamics(t, x, U, ctrl, sys)
    
    % state
    r = x(1);
    theta = x(2);
    rdot = x(3);
    thetadot = x(4);

    % input
    u = interpolate_input(t, U, ctrl, sys);
    if (r<= sys.r_min) && (rdot <= 0)
        u(1) = max(u(1), 0);
    end
    if (r >= sys.r_max) && (rdot >= 0)
        u(1) = min(u(1), 0);
    end

    if (theta <= sys.theta_min) && (thetadot <= 0)
        u(2) = max(u(2), 0);
    end
    if (theta >= sys.theta_max) && (thetadot >= 0)
        u(2) = min(u(2), 0);
    end
    
    % dynamics
    zeros_mat = zeros(2, 2);
    ones_mat = eye(2);
    A = [zeros_mat, ones_mat;
         zeros_mat, zeros_mat];
    B = [zeros(2, 2); eye(2)];
    xdot = A * x + B * u;

end

% interpolate the input
function u = interpolate_input(t, U, ctrl, sys)
    
    N = ctrl.N;
    dt = ctrl.dt;
    T = (0:N-1) * dt;

    % find the index
    idx = find(T <= t, 1, 'last');

    % linear interpolation
    if ctrl.interp == 'Z'
        u = U(:, idx);
    
    % zero-order hold
    elseif ctrl.interp == 'L'
        if idx == N
            u = zeros(2, 1);
        else
            u0 = U(:, idx);
            uf = U(:, idx+1);
            u = u0 + (uf - u0) * (t - T(idx)) / dt;
        end
    end
end

function u_clipped = saturate_input(u, sys)
    u_clipped = zeros(2, 1);
    u_clipped(1) = min(max(u(1), -sys.rdot_max), sys.rdot_max);
    u_clipped(2) = min(max(u(2), -sys.thetadot_max), sys.thetadot_max);
end

% sample a trajecotry
function U = sample_input(ctrl, distr, sys)

    % Uniform 
    if distr.type == 'U'    
        % upper and lower bounds
        A = ones(ctrl.N, 1);
        lb = distr.bounds(:, 1);
        ub = distr.bounds(:, 2);
        lb_vec = kron(A, lb);
        ub_vec = kron(A, ub);

        % sample
        U_vec = unifrnd(lb_vec, ub_vec);
    
    % Gaussian
    elseif distr.type == 'G'
        % mean
        ones_vec = ones(ctrl.N, 1);
        mean = kron(ones_vec, distr.mean);

        % covariance
        cov_diag = diag([distr.std_r^2, distr.std_theta^2]);
        I = eye(ctrl.N);
        cov = kron(I, cov_diag);

        % sample
        U_vec = mvnrnd(mean, cov)';
    end

    % pack into a matrix
    U = zeros(2, ctrl.N);
    for i = 1:ctrl.N
        ui = U_vec( 2*i-1 : 2*i );
        ui_clipped = saturate_input(ui, sys);
        U(:, i) = ui_clipped;
    end
end


