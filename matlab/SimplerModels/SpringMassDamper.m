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
mpc.K = 500;               % number of rollouts
mpc.N = 50;                % number of time steps
mpc.dt = 0.005;            % time step
mpc.interp = 'L';          % interpolation type
mpc.Q  = diag([6, 0.05]);  % state cost
mpc.Qf = diag([7, 0.05]);  % state cost
mpc.R = 0.0;              % input cost
mpc.Rr = 1e-3;              % input rate cost
mpc.iters = 25;            % number of iterations
mpc.n_elite = 10;          % number of elite rollouts

% distribution struct
distr.mu = 0.45 * ones(mpc.N-1, 1);             % mean
distr.u_max = 1.0;                               % maximum input
distr.u_min = 0.1;                               % minimum input
sigma = 1.0;                                    % standard deviation
distr.Sigma = diag(sigma^2 * ones(mpc.N-1, 1)); % covariance
distr.cov_min = 0.1;                            % keep the diagonals above this value

% simulation parameters
rt = 0.1; % real time rate
n_replays = 3;

% initial conditions
x0 = [0.2;    % inital position 
      0.0];   % initial velocity
xdes = [0.7; % desired position
        0.0]; % desired velocity
Xdes = generate_reference(x0, xdes, mpc);

% simulate the system 
[t, x, u] = cross_entropy_method(x0, Xdes, sys, mpc, distr);

animate(t, x, Xdes, u, mpc, rt, n_replays);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROLLOUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cross entropy method
function [t, x, u] = cross_entropy_method(x0, Xdes, sys, mpc, distr)

    % initialize the results
    for i = 1:mpc.iters

        % perform a monte carlo simulation
        [X, U, J] = monte_carlo(x0, Xdes, sys, mpc, distr);

        % find the best rollouts
        [~, idx] = sort(J, "ascend");
        idx_elite = idx(1:mpc.n_elite);
        U_elite = U(:, :, idx_elite);
        U_elite_matrix = reshape(U_elite, mpc.N-1, mpc.n_elite);

        % update the distribution
        U_mean = mean(U_elite_matrix, 2);
        U_cov = cov(U_elite_matrix');
        diags_cov = diag(U_cov);
        for j = 1:length(diags_cov)
            if diags_cov(j) < distr.cov_min
                diags_cov(j) = distr.cov_min;
            end
        end
        U_cov = diag(diags_cov);
        % [~, S, ~] = svd(U_cov);
        % diags_S = diag(S);
        % for j = 1:length(diags_S)
        %     if diags_S(j) < distr.cov_min
        %         diags_S(j) = distr.cov_min;
        %     end
        % end
        % U_cov = diag(diags_S);

        % update the distribution
        distr.mu = U_mean;
        distr.Sigma = U_cov;

        Sigma_norm = norm(distr.Sigma);
        iter = i;
        J_best = J(idx(1));
        disp(['Iteration: ', num2str(iter), ' | Best Cost: ', num2str(J_best), ' | Sigma Norm: ', num2str(Sigma_norm)]);
    end
    
    % find the best rollouts
    t = 0:mpc.dt:mpc.dt*(mpc.N-1);
    [~, ind] = sort(J);
    x = X(:, :, ind(1));
    u = U(:, :, ind(1));
end

% do a monte carlo simulation
function [X, U, J] = monte_carlo(x0, Xdes, sys, mpc, distr)

    % initialize the results
    X = zeros(mpc.N, 2, mpc.K);
    U = zeros(mpc.N-1, 1, mpc.K);
    
    % perform a rollout
    for k = 1:mpc.K
        % disp(['Rollout: ', num2str(k)]);
        [x_t, u_t] = rollout(x0, sys, mpc, distr);
        X(:, :, k) = x_t;
        U(:, :, k) = u_t;
    end

    % evaluate the cost of each rollout
    J = zeros(mpc.K, 1);
    for k = 1:mpc.K
        % disp(['Cost: ', num2str(k)]);
        J(k) = cost_function(X(:, :, k), U(:, :, k), Xdes, mpc);
    end
end

% perform a rollout
function [x_t, u_t] = rollout(x0, sys, mpc, distr)

    % generate a random input signal
    u_t = sample_input(mpc, distr);

    % generate the tpsan
    tspan = 0:mpc.dt:mpc.dt*(mpc.N-1);

    % simulate the system
    [~, x_t] = ode45(@(t, x) dynamics(t, x, u_t, sys, mpc), tspan, x0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COST FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function J = cost_function(x, u, Xdes, mpc)

    % soime params
    N = mpc.N;
    Q = mpc.Q;
    R = mpc.R;

    % integrated cost
    cost = 0;
    for i = 1:N-1
        
        xk = x(i, :)';
        xk_des = Xdes(i, :)';
        uk = u(i);

        state_cost = (xk - xk_des)' * Q * (xk - xk_des);
        input_cost = uk^2 * R;
    
        cost = cost + state_cost + input_cost;
    end

    % terminal cost
    xN = x(N, :)';
    xN_des = Xdes(N, :)';
    terminal_cost = (xN - xN_des)' * mpc.Qf * (xN - xN_des);

    % Penalty on the rate of change of the input
    U_rate = diff(u);
    input_rate_cost = norm(U_rate)^2 * mpc.Rr;
    
    J = state_cost + input_cost + terminal_cost + input_rate_cost;
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
function U = sample_input(mpc, distr)

    % sample a gaussian disitrbution
    U = mvnrnd(distr.mu, distr.Sigma)';

    % clip the values
    for i = 1:length(U)
        U(i) = max(min(U(i), distr.u_max), distr.u_min);
    end
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

% generate the reference trajectory
function Xdes = generate_reference(x0, xdes, mpc)

    % do linear interpolation as the reference
    Xdes(:, 1) = linspace(x0(1), xdes(1), mpc.N);

    v_const = (xdes(1) - x0(1)) / (mpc.dt * mpc.N);
    Xdes(:, 2) = v_const * ones(mpc.N, 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANIMATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function animate(t, x, xdes, u, mpc, rt, n_replays)

    % plot the results
    figure('Name', 'Spring-Mass-Damper Simulation');

    subplot(3, 2, 1); hold on;
    plot(t, x(:, 1), 'LineWidth', 2);
    plot(t, xdes(:, 1), 'k--', 'LineWidth', 2);
    xlabel('Time [s]');
    ylabel('Position [m]');
    grid on;

    subplot(3, 2, 3); hold on;
    plot(t, x(:, 2), 'LineWidth', 2);
    plot(t, xdes(:, 2), 'k--', 'LineWidth', 2);
    xlabel('Time [s]');
    ylabel('Velocity [m/s]');
    grid on;

    subplot(3, 2, 5);
    if mpc.interp == 'L'
        plot(t(1:end-1), u, 'LineWidth', 2);
    elseif mpc.interp == 'Z'
        stairs(t(1:end-1), u, 'LineWidth', 2);
    end
    ylabel('Free Length, l0 [m]');
    grid on;

    subplot(3, 2, [2,4,6]);
    hold on;
    yline(0); 
    ylabel('Position [m]');
    axis equal; grid on;
    z_max = max(x(:, 1));
    z_min = min(x(:, 1));
    xlim([-0.3, 0.3]);
    ylim([min(z_min, 0)-0.1, max(z_max, 0)+0.1]);

    t = t * (1/rt);
    
    for i = 1:n_replays

        % play the animation
        pause(0.25);
        tic;
        ind = 1;
        while 1==1

            % draw the goal
            x_goal = xdes(ind, 1);
            x_goal_line = yline(x_goal, 'k--', 'desired');

            % draw the mass spring damper system
            pole = plot([0, 0], [0, x(ind, 1)], 'k', 'LineWidth', 2);
            box_center = [0; x(ind, 1)];
            box = rectangle('Position', [box_center(1)-0.1, box_center(2)-0.1, 0.2, 0.2], 'Curvature', 0.1, 'FaceColor', 'r');
            
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
                delete(x_goal_line);
                delete(pole);
                delete(box);
            end
        end

        % delete the residual stuff
        if i < n_replays
            delete(x_goal_line);
            delete(pole);
            delete(box);
        end
    end
end
