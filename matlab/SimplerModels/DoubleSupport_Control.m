%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Double Support system with Sampling Predictive Control (SPC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% system parameters
params.m = 25.0;   % mass 
params.g = 9.81;   % gravity
params.l0 = 1.5;   % free length of the leg
params.k = 4000;  % spring constant
params.b = 10.0;    % damping coefficient
params.p1 = [0; 0]; % left leg position
params.p2 = [0.5; 0]; % right leg position

% SPC parameters
spc.K = 3000;  % number of rollouts
spc.dt = 0.02; % time step
spc.N = 25;    % prediction horizon
spc.Q = diag([10, 30, 0.1, 0.1]); % state cost
spc.R = diag([0.0, 0.0]);       % control cost
spc.Qf = diag([30, 50, 10, 10]); % final state cost
spc.n_elite = 1; % number of elite rollouts (Note: why does it suck with n =/= 1?)
spc.n_iters = 15; % number of CE-M iterations

% initial distribution parameters
distr.type = 'G'; % distribution type, 'G' (gaussian) or 'U' (uniform)
distr.mu = [0; 0];
distr.Sigma = diag([1000^2, 1000^2]);
distr.Unif = [-50; 50];

% simulation params
rt = 0.25;         % real time rate multiplier
animate = 1;

% intial conditions
x0 = [0.25;    % px
      1.45378; % pz
      0;   % vx
      0];  % vz

% desired state
x_des = [0.25; 
         1.0; 
         0; 
         0];

% monte carlo simulation
[t, x, u] = simulate(x0, x_des, params, spc, distr);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot the data
if animate == 1
    figure('Name', 'Animation', 'WindowState', 'maximized');
    set(0, 'DefaultFigureRenderer', 'painters');

    subplot(2, 3, 1);
    hold on;
    plot(t, x(:, 1), 'b', 'LineWidth', 1);
    yline(x_des(1), '--', 'Desired');
    xline(0, '--');
    yline(0, '--');
    xlabel('t'); ylabel('px');
    grid on;

    subplot(2, 3, 2);
    hold on;
    plot(t, x(:, 2), 'b', 'LineWidth', 1);
    yline(x_des(2), '--', 'Desired');
    xline(0, '--');
    yline(0, '--');
    xlabel('t'); ylabel('pz');
    grid on;

    subplot(2, 3, 4);
    hold on;
    plot(t, x(:, 3), 'b', 'LineWidth', 1);
    yline(x_des(3), '--', 'Desired');
    xline(0, '--');
    yline(0, '--');
    xlabel('t'); ylabel('vx');
    grid on;

    subplot(2, 3, 5);
    hold on;
    plot(t, x(:, 4), 'b', 'LineWidth', 1);
    yline(x_des(4), '--', 'Desired');
    xline(0, '--');
    yline(0, '--');
    xlabel('t'); ylabel('vz');
    grid on;

    subplot(2, 3, 6);
    hold on;
    plot(t(1:end-1), u(:, 1), 'b', 'LineWidth', 1);
    plot(t(1:end-1), u(:, 2), 'r', 'LineWidth', 1);
    xline(0, '--');
    xlabel('t'); ylabel('u');
    legend('u1', 'u2');
    grid on;

    subplot(2, 3, 3);
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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% double support dynamics
function xdot = dynamics(x, u, params)

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

    % unpack control
    u1 = u(1);
    u2 = u(2);

    % compute the leg vectors
    r1 = p_com - p1;
    r2 = p_com - p2;
    r1_norm = norm(r1);
    r2_norm = norm(r2);
    r1_hat = r1/r1_norm;
    r2_hat = r2/r2_norm;

    % compute the dynamics
    a_com = r1_hat * ((k/m) * (l0 - r1_norm) - (b/m) * (v_com' * r1) / r1_norm + (1/m) * u1) ...
          + r2_hat * ((k/m) * (l0 - r2_norm) - (b/m) * (v_com' * r2) / r2_norm + (1/m) * u2) ...
          + [0; -g];
    xdot = [v_com; a_com];

end

% custom RK3 integrator
function [t, x] = RK3_rollout(x0, u, params, spc)

    % integration parameters
    dt = spc.dt;
    N = spc.N;
    
    % make containers
    t = 0:dt:dt*(N-1);
    x = zeros(N, length(x0));
    x(1, :) = x0;
    xk = x0;

    % RK3 integration
    for k = 1:N-1

        % compute the control input
        uk = u(k, :);

        % RK3 integration
        f1 = dynamics(xk, uk, params);
        f2 = dynamics(xk + 0.5*dt*f1, uk, params);
        f3 = dynamics(xk - dt*f1 + 2*dt*f2, uk, params);
        xk = xk + (dt/6)*(f1 + 4*f2 + f3);

        % store the state
        x(k+1, :) = xk';
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [t, x_star, u_star] = simulate(x0, xdes, params, spc, distr)

    % unpack the SPC parameters
    n_elite = spc.n_elite;
    n_iters = spc.n_iters;

    for i = 1:n_iters

        % display some info
        disp('-----------------------------------');
        fprintf('Iteration: %d \n', i);
        fprintf('Mean: ');
        distr.mu
        fprintf('Sigma: ');
        distr.Sigma

        % monte carlo simulation of K trajectories
        [X, U, J] = monte_carlo(x0, xdes, params, spc, distr);

        % sort from lowest cost to highest
        [~, idx] = sort(J, "ascend");
        U = U(:, :, idx);

        % pick the top elite input samples
        u_elite = U(:, :, 1:n_elite);
        u_mean = mean(u_elite, 3);
        
        % update the parametric distribution
        distr.mu = mean(u_mean)';
        distr.Sigma = cov(u_mean);
    end

    % stuff to return
    t = 0:spc.dt:spc.dt*(spc.N-1);
    X = X(:, :, idx);
    U = U(:, :, idx);
    x_star = X(:, :, 1);
    u_star = U(:, :, 1);
end

% monte carlo simualtion
function [X, U, J] = monte_carlo(x0, xdes, params, spc, distr)

    % mpc parameters
    N = spc.N;
    K = spc.K;

    % make containers
    X = zeros(N, 4, K);   % store the state trajectories
    U = zeros(N-1, 2, K); % store the control trajectories
    J = zeros(K, 1);      % store the costs

    % simualte a rollout
    for k = 1:K

        % sample the input
        u = sample_input(spc, distr);

        % perform a dynamics rollout
        [t, x] = RK3_rollout(x0, u, params, spc);

        % compute the cost
        c = eval_cost(xdes, x, u, spc);

        % store the data
        X(:, :, k) = x;
        U(:, :, k) = u;
        J(k) = c;
    end
end

% comptue cost
function J = eval_cost(xdes, x, u, spc)

    % unpack the parameters
    Q = spc.Q;
    R = spc.R;
    Qf = spc.Qf;
    N = spc.N;

    % compute the state cost
    % NOTE: cost can be anything
    stage_cost = 0;
    for i = 1:N-1
        % compute the state cost
        state_cost = (x(i, :)' - xdes)' * Q * (x(i, :)' - xdes);
        
        % control the control cost
        control_cost = u(i, :) * R * u(i, :)';

        % accumulate the stage cost
        stage_cost = state_cost + control_cost + stage_cost;
    end

    % compute the final state cost
    terminal_cost = (x(N, :)' - xdes)' * Qf * (x(N, :)' - xdes);

    % total cost
    J = stage_cost + terminal_cost;
end

% sample an input given a disitrbution
function u = sample_input(spc, distr)

    % sample the input
    if distr.type == 'U'
        u = unifrnd(distr.Unif(1), distr.Unif(2), spc.N-1, 2);
    elseif distr.type == 'G'
        u = mvnrnd(distr.mu, distr.Sigma, spc.N-1);
    end
end
