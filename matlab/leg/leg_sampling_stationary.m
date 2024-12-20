%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample the leg trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc; 

% generate random bezier control points
deg = 5; % polynomial degree
dim = 3; % dimension

lb = -1;
ub = 1;

[T, B, P] = generate_trajectory(dim, deg, lb, ub);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute the binomial coefficients of bezier curve
function c = bezier_coeff(n , k)
    c = factorial(n) / (factorial(k) * factorial(n - k));
end

% genearte random trajectory
function [T, B, P] = generate_trajectory(dim ,deg, lb, ub)
    
    % generate random control points
    T = linspace(0, 1, 100);
    B = zeros(dim, length(T));

    % generate random control points
    P = unifrnd(lb, ub, dim, deg + 1);

    % compute the bezier curve
    for i = 0:deg
        Pi = P(:, i + 1);
        ci = bezier_coeff(deg, i);
        term = ci * Pi * (1 - T).^(deg - i) .* T.^i;
        B = B + term;
    end

end