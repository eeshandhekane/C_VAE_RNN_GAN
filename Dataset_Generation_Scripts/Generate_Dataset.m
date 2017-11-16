function [ ] = Generate_Dataset( )
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
% Generate all the datasets
velocity = [10, 15, 20, 25, 30, 35, 40];
inclination = [7, 15, 22, 30, 37, 45, 60, 75];
dt = 0.1;
restitution = [1];
gravity = [10];
arch = 5;
seq_len = 60;
blob_size = [10];
num = 5;
%
% Generate in loop the dataset
for i = 1:size(velocity, 2)
    v = velocity(i);
    for j = 1:size(inclination, 2)
        theta = inclination(j);
        for k = 1:size(restitution, 2)
            e_x = restitution(k);
            e_y = e_x;
            for l = 1:size(gravity, 2)
                g = gravity(l);
                for m = 1:size(blob_size, 2)
                    blob = blob_size(m);
                    % Generate the points of the projectile
                    points = Get_Projectile_Points( v, theta, dt, e_x, e_y, g, arch );
                    points_size = size(points);
                    if points_size(1, 1) >= 125
                        % If sufficiently many images exist, create data
                        Generate_Image_Sequence( v, theta, dt, e_x, e_y, g, arch, seq_len, blob, num, points )
                    end
                end
            end
        end
    end
end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
