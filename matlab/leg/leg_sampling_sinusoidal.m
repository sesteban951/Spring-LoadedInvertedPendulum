%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample the leg trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc; 

% system parameters
sys.r_nom = 0.65;
sys.r_min = 0.40;
sys.r_max = 0.80;
sys.theta_nom = 0.0;
sys.theta_min = -pi/4;
sys.theta_max = pi/4;

% control parameters
ctrl.N = 150;             % Number of knots
ctrl.dt = 0.02;          % Time step (seconds)
ctrl.n_harmonics = 2;    % Number of harmonics
ctrl.f0 = 1.0;            % Fundamental frequency (Hz)

% distribution parameters
distr.type = 'G';
distr.bounds = [-1.0, 1.0;     
                -1.0, 1.0];  
distr.mean = [0.0;
              0.0];
distr.std_r = 1.5;
distr.std_theta = 1.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sample the leg trajectories
samples = 10;
figure(1);
for i = 1:samples
    t = linspace(0, ctrl.N*ctrl.dt, ctrl.N); % Time vector
    U = sample_fourier(sys, ctrl, distr);

    % post process
    px = -U(1, :) .* sin(U(2, :));
    pz = -U(1, :) .* cos(U(2, :));
    theta_deg = rad2deg(U(2, :));
    subplot(2, 2, 1)
    plot(t, U(1, :), 'b', 'LineWidth', 2);
    yline(0)
    xlabel('Time (s)');
    ylabel('Length (m)');
    grid on;

    subplot(2, 2, 3)  
    plot(t, U(2, :), 'r', 'LineWidth', 2);
    yline(0)
    xlabel('Time (s)');
    ylabel('Angle (rad)');
    grid on;

    subplot(2, 2, [2, 4])
    hold on; grid on; axis equal;
    xline(0)
    yline(0)
    plot(0, 0, 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k');
    x_min = min(min(px), 0) - 0.1;
    x_max = max(max(px), 0) + 0.1;
    z_min = min(min(pz), 0) - 0.1;
    z_max = max(max(pz), 0) + 0.1;
    xlim([x_min, x_max]);
    ylim([z_min, z_max]);

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

    % clear the plots
    if i < samples
        clf
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function U = sample_fourier(sys, ctrl, distr)

    % system parameters
    r_max = sys.r_max;
    r_min = sys.r_min;
    theta_max = sys.theta_max;
    theta_min = sys.theta_min;

    % control parameters
    dt = ctrl.dt;          % Time step (seconds)
    N = ctrl.N;            % Number of knots
    t = linspace(0, N*dt, N); % Time vector

    % Fourier series parameters
    n = ctrl.n_harmonics;  % Number of harmonics
    f0 = ctrl.f0;          % Fundamental frequency (Hz)

    % Generate random coefficients
    a_n = 0.05 * randn(1, N);     % Random cosine amplitudes (normal distribution)
    b_n = 0.05 * randn(1, N);     % Random sine amplitudes (normal distribution)

    % DC offset
    c0 = sys.r_nom;

    % Construct the Fourier series
    u_r = c0 * ones(size(t));   % Initialize signal with the DC component
    for i = 1:n
        u_r = u_r + a_n(i) * cos(2 * pi * i * f0 * t) + ...
                    b_n(i) * sin(2 * pi * i * f0 * t);
    end

    % Generate random coefficients
    a_n = 0.1 * randn(1, N);     % Random cosine amplitudes (normal distribution)
    b_n = 0.1 * randn(1, N);     % Random sine amplitudes (normal distribution)

    % DC offset
    c0 = sys.theta_nom;

    % Construct the Fourier series
    u_theta = c0 * ones(size(t));   % Initialize signal with the DC component
    for i = 1:n
        u_theta = u_theta + a_n(i) * cos(2 * pi * i * f0 * t) + ...
                            b_n(i) * sin(2 * pi * i * f0 * t);
    end

    % saturate the signals
    u_r = min(r_max, max(r_min, u_r));
    u_theta = min(theta_max, max(theta_min, u_theta));

    U = [u_r; u_theta];
end