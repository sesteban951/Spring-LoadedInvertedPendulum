%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLot some sim data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load data
t = load('./data/time.csv');
x_com = load('./data/state_com.csv');
x_left = load('./data/state_left.csv');
x_right = load('./data/state_right.csv');
p_left = load('./data/pos_left.csv');
p_right = load('./data/pos_right.csv');
fileID = fopen('./data/domain.csv', 'r');
domain = textscan(fileID, '%s', 'Delimiter', ',');
domain = char(domain{1});
fclose(fileID);

% COM state
p_com = x_com(:,1:2);
v_com = x_com(:,3:4);

% leg states
r_left = x_left(:,1:2);
r_right = x_right(:,1:2);
theta_left = x_left(:,3);
theta_right = x_right(:,3);

% animation params
rt = 1.0; % realtime rate
animate = true;

% animate the com trajectory
if animate == true

    figure('Name', 'Animation');
    set(0, 'DefaultFigureRenderer', 'painters');
    hold on;

    xline(0);
    yline(0);
    xlabel('$p_x$ [m]', 'Interpreter', 'latex');
    ylabel('$p_z$ [m]', 'Interpreter', 'latex');
    grid on; axis equal;
    px_min = min([p_com(:,1); p_left(:,1); p_right(:,1)]);
    px_max = max([p_com(:,1); p_left(:,1); p_right(:,1)]);
    pz_min = min([p_com(:,2); p_left(:,2); p_right(:,2)]);
    pz_max = max([p_com(:,2); p_left(:,2); p_right(:,2)]);
    xlim([px_min-0.25, px_max+0.25]);
    ylim([min(0, pz_min)-0.25, pz_max+0.25]);
    
    t  = t * (1/rt);
    pause(0.5);
    tic;
    ind = 1;
    while true

        % get COM position 
        px = p_com(ind,1);
        pz = p_com(ind,2);

        % draw the legs
        d = domain(ind,:);
        if d ~= 'F'
            if d == 'L'
                px_left = p_left(ind,1);
                pz_left = p_left(ind,2);
                
                left_leg = plot([px, px_left], [pz, pz_left], 'b', 'LineWidth', 2);
                left_foot = plot(px_left, pz_left, 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
                right_leg = plot(NaN, NaN);
            elseif d == 'R'
                px_right = p_right(ind,1);
                pz_right = p_right(ind,2);
                
                left_leg = plot(NaN, NaN);
                right_leg = plot([px, px_right], [pz, pz_right], 'r', 'LineWidth', 2);
                right_foot = plot(px_right, pz_right, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
            elseif d == 'D'
                px_left = p_left(ind,1);
                pz_left = p_left(ind,2);
                px_right = p_right(ind,1);
                pz_right = p_right(ind,2);

                left_leg = plot([px, px_left], [pz, pz_left], 'b', 'LineWidth', 2);
                right_leg = plot([px, px_right], [pz, pz_right], 'r', 'LineWidth', 2);
                left_foot = plot(px_left, pz_left, 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
                right_foot = plot(px_right, pz_right, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
            end
        else
            left_leg = plot(NaN, NaN);
            right_leg = plot(NaN, NaN);
        end

        % draw the mass
        mass = plot(px, pz, 'ko', 'MarkerSize', 25, 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
        pt_pos = plot(px, pz, 'k.', 'MarkerSize', 5);

        drawnow;
        
        % title
        msg = sprintf('Time: %0.3f [sec]', t(ind) * rt);
        title(msg);
        
        % wait until the next time step
        while toc< t(ind+1)
            % wait
        end
        
        % increment the index
        if ind+1 >= length(t)
            break;
        else
            ind = ind + 1;
            delete(mass);
            delete(left_leg);
            delete(right_leg);
        end
    end
end