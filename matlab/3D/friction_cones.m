clear all; clc; close all;
% Define the friction coefficient mu
mu = 0.5;

% pick a maximum level
z_max = 100;

% evaluate the friction cone four corners
x_max = z_max * mu;
x_min = -z_max * mu;
y_max = z_max * mu;
y_min = -z_max * mu;

% get the four corners of the friction cone
p1 = [x_max; y_max; z_max];
p2 = [x_max; y_min; z_max];
p3 = [x_min; y_min; z_max];
p4 = [x_min; y_max; z_max];

% add the origin
p5 = [0; 0; 0];

%  all the points to create lines between
P = [p1'; p2'; p3'; p4'; p5'];

% plot the friction cone
figure;
grid on; hold on; axis equal;

% plot convex hull
P_T = P;
[K, ~] = convhull(P_T);
trisurf(K, P_T(:, 1), P_T(:, 2), P_T(:, 3), 'FaceColor', 'g', 'FaceAlpha', 0.5);
alpha(0.1);

% plot the lines
for i = 1:5
    for j = 1:5
        if i ~= j
            plot3([P(i, 1), P(j, 1)], [P(i, 2), P(j, 2)], [P(i, 3), P(j, 3)], 'g', 'LineWidth', 2);
        end
    end
end

% Set the view angle
view([30, 35]);