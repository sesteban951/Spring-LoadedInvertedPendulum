%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Double Support system with CE-M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% system parameters
params.m = 25.0;   % mass 
params.g = 9.81;   % gravity
params.l0 = 1.5;   % free length of the leg
params.k = 4000;  % spring constant
params.b = 400.0;    % damping coefficient
params.p1 = [0; 0]; % left leg position
params.p2 = [0.5; 0]; % right leg position

% SPC parameters
spc.K = 1000;  % number of rollouts
spc.dt = 0.01; % time step
spc.N = 50;    % prediction horizon
spc.Q = diag([30, 30, 0.1, 0.1]); % state cost
spc.R = diag([0., 0.]);           % control cost
spc.Qf = diag([50, 50, 1, 1]);    % final state cost
spc.n_elite = 10;      % number of elite rollouts to consider
spc.n_iters = 15;      % number of CE-M iterations
spc.cov_scaling = 2.0; % scaling factor for the covariance to not collapse too fast

% initial distribution parameters
distr.type = 'G'; % distribution type to use, 'G' (gaussian) or 'U' (uniform)
distr.mu = [params.l0; params.l0];
distr.Sigma = diag([2.0^2, 2.0^2]);
distr.Unif = [params.l0 - 1.5, params.l0 + 1.5];

% simulation params
rt = 0.25;         % real time rate multiplier
animate = 1;
plot_pdf = 1;

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
[t, x, u, distr_history] = simulate(x0, x_des, params, spc, distr);

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
    stairs(t(1:end-1), u(:, 1), 'b', 'LineWidth', 1);
    stairs(t(1:end-1), u(:, 2), 'r', 'LineWidth', 1);
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
        mass = plot(px, pz, 'ko', 'MarkerSize', 20, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);

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

% plot the distribution history
if plot_pdf == 1

    pause(1);

    if distr.type == 'G' 
        % for plotting the dsitribution
        figure('Name', 'Distribution History');
        hold on;

        % make a meshgrid
        mu_final = distr_history.Mu(end, :);
        mu_max = max(mu_final) + 1;
        mu_min = min(mu_final) - 1;
        [X,Y] = meshgrid(mu_min:0.02:mu_max);
        xlim([mu_min, mu_max]);
        ylim([mu_min, mu_max]);

        % plot the distribution history
        for i = 1:spc.n_iters
            
            % get the current distribution parameters
            mu = distr_history.Mu(i, :);
            Sigma = distr_history.Sigma(:, :, i);

            % plot the distribution
            pdf = mvnpdf([X(:) Y(:)], mu, Sigma);
            pdf = reshape(pdf, size(X));
            pdf_plot = pcolor(X, Y, pdf); 

            % plot the mean
            mean_pt = plot(mu(1), mu(2), 'r.', 'MarkerSize', 10);
            shading interp;

            % tile
            msg = sprintf('Iteration: %d\nmu = [%f, %f]\nSigma = [%f, %f; %f, %f]', i, mu(1), mu(2), Sigma(1, 1), Sigma(1, 2), Sigma(2, 1), Sigma(2, 2));
            title(msg);

            if i < spc.n_iters
                pause(0.75);
                delete(pdf_plot);
                delete(mean_pt);
            end
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

    % force inputs
    % u1 = u(1);
    % u2 = u(2);

    % free length inputs
    v1 = u(1);
    v2 = u(2);
    u1 = k * (v1 - l0);
    u2 = k * (v2 - l0);

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

function [t, x_star, u_star, distr_history] = simulate(x0, xdes, params, spc, distr)

    % unpack the SPC parameters
    n_elite = spc.n_elite;
    n_iters = spc.n_iters;

    % containers to store the distribution history
    if distr.type == 'U'
        Unif = zeros(n_iters, 2);
    elseif distr.type == 'G'
        Mu = zeros(n_iters, 2);
        Sigma = zeros(2, 2, n_iters);
    end

    % converge on the optimal disitrubtion
    for i = 1:n_iters

        % display some info
        disp('-----------------------------------');
        fprintf('Iteration: %d \n', i);
        
        if distr.type == 'U'
            fprintf('Uniform Range: [%f, %f] \n', distr.Unif(1), distr.Unif(2));
        elseif distr.type == 'G'
            fprintf('Normal Mean: ');
            distr.mu
            fprintf('Normal Sigma: ');
            distr.Sigma
        end

        % monte carlo simulation of K trajectories
        [X, U, J] = monte_carlo(x0, xdes, params, spc, distr);

        % sort from lowest cost to highest
        [~, idx] = sort(J, "ascend");
        U = U(:, :, idx);

        % pick the top elite input samples
        u_elite = U(:, :, 1:n_elite);
        u_mean = mean(u_elite, 3);  % averaging the n elite samples
                                    % getting an average input tape

        % update the parametric distribution
        if distr.type == 'U'

            % save the uniform distribution
            Unif(i, :) = distr.Unif;

            % update the uniform distribution
            distr.Unif = [min(u_mean(:)), max(u_mean(:))]; % need better way to update the UNiform parameters

        elseif distr.type == 'G'
            
            % save the gaussian distribution
            Mu(i, :) = distr.mu';
            Sigma(:, :, i) = distr.Sigma;

            % update the gaussian dsitribution
            distr.mu = mean(u_mean)';
            distr.Sigma = spc.cov_scaling * cov(u_mean);
        end
    end

    % stuff to return
    t = 0:spc.dt:spc.dt*(spc.N-1);
    X = X(:, :, idx);
    U = U(:, :, idx);
    x_star = X(:, :, 1);
    u_star = U(:, :, 1);

    % store the distribution history
    if distr.type == 'U'
        distr_history = Unif;
    elseif distr.type == 'G'
        distr_history.Mu = Mu;
        distr_history.Sigma = Sigma;
    end
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
        c = eval_cost(xdes, x, u, params, spc);

        % store the data
        X(:, :, k) = x;
        U(:, :, k) = u;
        J(k) = c;
    end
end

% comptue cost
function J = eval_cost(xdes, x, u, params, spc)

    % unpack the parameters
    Q = spc.Q;
    R = spc.R;
    Qf = spc.Qf;
    N = spc.N;
    l0_nom = [params.l0, params.l0];

    % compute the state cost
    % NOTE: cost can be anything
    stage_cost = 0;
    for i = 1:N-1
        % compute the state cost
        state_cost = (x(i, :)' - xdes)' * Q * (x(i, :)' - xdes);
        
        % control the control cost
        control_cost = (u(i, :) - l0_nom) * R * (u(i, :) - l0_nom)';

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
