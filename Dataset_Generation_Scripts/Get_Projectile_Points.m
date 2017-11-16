function [ proj_points ] = Get_Projectile_Points( v, theta, dt, e_x, e_y, g, arch )
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUTS--
%
% OUTPUTS--
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Initialise loop variables
v_x = v*cos(deg2rad(theta));
v_y = v*sin(deg2rad(theta));
arch_count = 1;
proj_points = [0, 0];
%
% Generate 10 arches of projectile (arch has default input value 11)
while arch_count ~= arch
    % Calculate the next position of the point
    curr_pos = proj_points(end, :);
    pred_disp = [v_x*dt, v_y*dt - 1/2*g*dt*dt]; 
    next_pos = curr_pos + pred_disp;
    % If valid, append the position to the list
    if next_pos(1, 2) > 0
        proj_points = [proj_points; next_pos];
        v_y = v_y - g*dt;
    % If invalid, increase the arch number and calculate the appropriate
    % position
    else
        % Increase the arch count
        arch_count = arch_count + 1;
        % Calculate the time of impact with ground
        coll_t = (v_y + (v_y*v_y + 2*g*curr_pos(1, 2))^(1/2))/(g);
        coll_x = curr_pos(1, 1) + v_x*coll_t;
        coll_y = 0;
        % Calculate the slowing down of the x-component of velocity
        v_x = v_x*e_x;
        v_y = abs(v_y - g*coll_t)*e_y;
        % Calculate the new position reached in the remaining time
        t_rem = dt - coll_t;
        new_pos = [coll_x + t_rem*v_x, coll_y + v_y*t_rem - 1/2*g*t_rem*t_rem];
        proj_points = [proj_points; new_pos];
        % Update velocities
        v_y = v_y - g*t_rem;
    end
end
%
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
