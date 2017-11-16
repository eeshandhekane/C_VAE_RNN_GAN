function [ ] = Generate_Image_Sequence( v, theta, dt, e_x, e_y, g, arch, seq_len, blob, num, point_list )
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
% Calculate the length of the point list
list_size = size(point_list);
len = list_size(1, 1);
% Select the initial point
init_range = 1: len - seq_len + 1;
ind = 1;
for i = 1: num
    init = datasample(init_range, 1) - 1;
    % Create a directory
    % v_theta_dt_ex_ey_g_arch_seqlen_{init+1}_blob_ind
    dir_name = strcat(num2str(v), '_', num2str(theta), '_', num2str(dt), '_', num2str(e_x), '_', num2str(e_y), '_', num2str(g), '_', num2str(arch), '_', num2str(seq_len), '_', num2str(blob), '_', num2str(init), '_', num2str(ind));
    mkdir(dir_name);
    im_num = 1;
    % Reference x
    x_ref = point_list(init + 1, 1);
    for j = 1:seq_len
        % Image name
        im_name = strcat(dir_name, '/', num2str(im_num), '.png');
        % Create the image by appropriately shifting the points
        arr = zeros(64, 64);
        x = point_list(init + im_num, 1);
        y = point_list(init + im_num, 2);
        x = x - x_ref;
        x = max(round(x)+1, 1);
        y = max(round(y)+1, 1);
        arr(x: x+blob, y: y+blob) = 1;
        % For correct orientation of the image, use transpose followed by flip
        new_size = size(arr);
        if new_size(1, 1) == 64 && new_size(1, 2) == 64
            imwrite(flipud(arr'), im_name);
        else
            crop_image = arr(1:64, 1:64);
            imwrite(flipud(crop_image'), im_name);
        end    
        % Increment to next image
        im_num = im_num + 1;
    end
    ind = ind + 1;
end
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
