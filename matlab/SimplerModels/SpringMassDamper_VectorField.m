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

% meshgrid for the state space
x1_lims = [-0.5, 0.5];
x2_lims = [-0.5, 0.5];
x1_mesh = linspace(x1_lims(1), x1_lims(2), 100);
x2_mesh = linspace(x1_lims(1), x1_lims(2), 100);
[x1, x2] = meshgrid(x1_mesh, x2_mesh);

% vector field
u = 0.15;
for i = 1:size(x1, 1)
    for j = 1:size(x1, 2)
        x = [x1(i, j); x2(i, j)];
        xdot = dynamics(0, x, u, sys);
        x1dot(i, j) = xdot(1);
        x2dot(i, j) = xdot(2);
    end
end

% plot the vector field
figure; hold on;
xline(u, 'r--', 'Input');
xline(0, 'k-');
yline(0, 'k-');
quiver(x1, x2, x1dot, x2dot, 'LineWidth', 2);
xlabel('Position [m]');
ylabel('Velocity [m/s]');
xlim([min(x1_mesh), max(x1_mesh)]);
ylim([min(x2_mesh), max(x2_mesh)]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMICS AND CONTROL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dynamics of the linear system
function xdot = dynamics(t, x, v, sys)

    % system parameters
    k = sys.k;
    m = sys.m;
    b = sys.b;
    g = sys.g;

    % convert the input to leg
    u = control(v, sys);

    % define the linear system matrices
    A = [0, 1; 
         -k/m, -b/m];
    B = [0; 
         1/m];
    C = [0; 
         (k/m)*sys.l0-g];

    % define the dynamics
    xdot = A*x + B*u + C;
end

function u = control(v, sys)
    u = sys.k * (v - sys.l0);
end
