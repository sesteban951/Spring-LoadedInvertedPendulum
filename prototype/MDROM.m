%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multidomain Reduced Order Model (MDROM) simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% system parameters
params.m = 35.0;   % mass (mass of Unitree G1)
params.g = 9.81;   % gravity
params.l0 = 0.65;   % Unitree G1 when standing is at 
                     % p_com_z at 0.707m (pelvis frame at 0.79m)
params.k = 4000;  % spring constant
params.b = 400.0;    % damping coefficient

% SPC parameters
spc.K = 1000;  % number of rollouts
spc.dt = 0.05; % time step
spc.N = 50;    % prediction horizon
spc.interp = 'L'; % interpolation method 'L' (linear) or 'Z' (zero order hold)

% initial distribution parameters
distr.type = 'U'; % distribution type to use, 'G' (gaussian) or 'U' (uniform)
distr.mu = [params.l0; % left leg length
            params.l0; % right leg length
            0;         % left angle
            0];        % right angle
distr.Sigma = diag([1.0^2;     % [m^2] left leg length
                    1.0^2;     % [m^2] right leg length
                    1.57^2;    % [rad^2] left angle
                    1.57^2]'); % [rad^2] right angle
distr.Unif = [params.l0 - 1.0, params.l0 + 1.0;  % left leg length
              params.l0 - 1.0, params.l0 + 1.0;  % right leg length
              -pi/2, pi/2;                       % left angle
              -pi/2, pi/2];                      % right angle

% initial state and domain
d = 'F';    % initial domain
x0 = [0;   % px
      0.7; % pz
      0;   % vx
      0];  % vz

U = sample_input(spc, distr);
T = 0:0.0001:3.0;

u_traj = zeros(4, length(T));
for i = 1:length(T)
    u_traj(:,i) = interpolate_input(T(i), U, spc);
end

plot(T, u_traj(1,:), 'r', 'LineWidth', 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% main dynamics function
function xdot = dynamics(t, x, u, d, p_feet, params)

    % extract the system parameters
    m = params.m;
    g = params.g;
    l0 = params.l0;
    k = params.k;
    b = params.b;
    
    % unpack state
    p_com = [x(1); x(2)];
    v_com = [x(3); x(4)];

    % Flight (F) phase
    if d == 'F'
        a_com = [0; -g];
        xdot = [v_com; a_com];

    % Left (L) leg on ground
    elseif d == 'L'
        
        % compute the leg state
        pL = [p_feet(1); p_feet(2)];
        rL = pL - p_com;
        rL_norm = norm(rL);
        rL_hat = rL / rL_norm;

        % get the control input
        vL = u(1);
        uL = k * (vL - l0);

        % compute the dynamics
        a_com = rL_hat * ((k/m) * (l0 - rL_norm) - (b/m) * (v_com' * rL) / rL_norm + (1/m) * uL) ...
                + [0; -g];
        xdot = [v_com; a_com];

    % Right (R) leg on ground
    elseif d == 'R'

        % compute the leg state
        pR = [p_feet(3); p_feet(4)];
        rR = pR - p_com;
        rR_norm = norm(rR);
        rR_hat = rR / rR_norm;

        % get the control input
        vR = u(2);
        uR = k * (vR - l0);

        % compute the dynamics
        a_com = rR_hat * ((k/m) * (l0 - rR_norm) - (b/m) * (v_com' * rR) / rR_norm + (1/m) * uR) ...
                + [0; -g];
        xdot = [v_com; a_com];

    % Double (D) support phase
    elseif d == 'D'
        
        % compute the leg state
        pL = [p_feet(1); p_feet(2)];
        pR = [p_feet(3); p_feet(4)];
        rL = pL - p_com;
        rR = pR - p_com;
        rL_norm = norm(rL);
        rR_norm = norm(rR);
        rL_hat = rL / rL_norm;
        rR_hat = rR / rR_norm;

        % get the control input
        vL = u(1);
        vR = u(2);
        uL = k * (vL - l0);
        uR = k * (vR - l0);

        % compute the dynamics
        a_com = rL_hat * ((k/m) * (l0 - rL_norm) - (b/m) * (v_com' * rL) / rL_norm + (1/m) * uL) ...
              + rR_hat * ((k/m) * (l0 - rR_norm) - (b/m) * (v_com' * rR) / rR_norm + (1/m) * uR) ...
              + [0; -g];
        xdot = [v_com; a_com];  
    end

end

% sample an input trajectory
function U = sample_input(spc, distr)

    % sample the input
    if distr.type == 'U'
        
        % sample from uniform distribution
        bounds = distr.Unif;
        uL = unifrnd(bounds(1,1), bounds(1,2), 1, spc.N-1);
        uR = unifrnd(bounds(2,1), bounds(2,2), 1, spc.N-1);
        thetaL = unifrnd(bounds(3,1), bounds(3,2), 1, spc.N-1);
        thetaR = unifrnd(bounds(4,1), bounds(4,2), 1, spc.N-1);
        U = [uL; uR; thetaL; thetaR];

    elseif distr.type == 'G'
        
        % sample from gaussian distribution
        mu = distr.mu;
        Sigma = distr.Sigma;
        U = mvnrnd(mu, Sigma, spc.N-1)';

    end
end

% interpolate the input trajectory
function u = interpolate_input(t, U, spc)

    % build the time array for the trajectory
    T_u = 0:spc.dt:spc.dt*(spc.N-2);

    % find where the time is in the trajectory
    idx = find(T_u <= t, 1, 'last');

    % zero order hold
    if spc.interp =='Z'
        % just constant input
        u = U(:,idx);

    % linear interpolation
    elseif spc.interp == 'L'
        % beyond the last point
        if idx == size(U,2)
            u = U(:,end);
        % linear interpolation
        else
            t1 = T_u(idx);
            t2 = T_u(idx+1);
            u1 = U(:,idx);
            u2 = U(:,idx+1);
            u = u1 + (u2 - u1) * (t - t1) / (t2 - t1);
        end
    end
end
