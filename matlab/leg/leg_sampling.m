clear all; close all; clc; 

% sample 
r = 1.0;

% unformly sample angles
N = 1000;

% for uniform sampling
lb = [-45, -45]; % lower bound
ub = [45, 45];   % upper bound
angles = [unifrnd(lb(1), ub(1), 1, N);
unifrnd(lb(2), ub(2), 1, N)]';

% for normal sampling
mu = [0; 0];            % mean
sigma = diag([45^2, 45^2]); % covaraince
angles = mvnrnd(mu, sigma, N);

% convert to radians
angles = angles * pi/180;        % rad 

% compute the foot positions
l_zero = [0; 0; -r];
l_final = zeros(N, 3);
for i = 1:N
    phi = angles(i, 1);
    psi = angles(i, 2);
    
    R_phi = [1, 0, 0;
             0, cos(phi), -sin(phi);
             0, sin(phi), cos(phi)];
    R_psi = [cos(psi),  0, sin(psi);
             0,         1, 0;
             -sin(psi), 0, cos(psi)];
    
    l_final(i, :) = R_psi * R_phi * l_zero;
end

% plot the unit sphere
figure;
hold on;
axis equal;

% plot the x, y, z axes
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);

% sphere
[X_sphere, Y_sphere, Z_sphere] = sphere;
surf(X_sphere, Y_sphere, Z_sphere, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.1);

% plot the leg positions
plot3([0, 0], [0, 0], [0, 0], 'k.', 'MarkerSize', 20);
plot3([0, 0], [0, 0], [0, -r], 'k--', 'LineWidth', 2);
plot3(l_final(:, 1), l_final(:, 2), l_final(:, 3), 'r.', 'MarkerSize', 10);

% animate the rotation
for i = 1:1
    for az = 0:360
        view(az, 30); % rotate around the z-axis
        pause(0.03); % pause to control the speed of rotation
    end
end
