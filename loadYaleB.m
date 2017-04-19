function [im, im_labels] = loadYaleB(options)
% Load the yale B dataface with 39 subjects and 64 lighting conditions.
% Image size is 192x168.
if isfield(options, 'len'), len = options.len;
else len = 80; end;
if isfield(options, 'wid'), wid = options.wid;
else wid = 60; end;
n_im_per_person = 64;
n_subject = 38;
subject_id = [1:13 15:39];
im = zeros(len*wid, n_im_per_person*n_subject);
im_labels = zeros(1, n_im_per_person*n_subject);
im_folder = '../yaleb/';
tot_valid_image = 0;
for i = 1:38
    i_sub = subject_id(i);
    im_subfolder = sprintf('%syaleB%02d/', im_folder, i_sub);
    D = dir(im_subfolder);
    n_valid_image = 0;
    for j_image = 1:length(D)
        im_name_str = D(j_image).name;
        valid_image_flag = strfind(im_name_str,'.pgm');
        im_subject = zeros(192, 168, n_im_per_person);
        % will return a valid image flag; non-empty if is an actual image
        if ~isempty(valid_image_flag)
            % this is a valid image
            % Check for ambient flags
            ambient_flag = strfind(im_name_str, 'Ambient');
            if isempty(ambient_flag)
                % This is an actual image of human faces
                n_valid_image = n_valid_image + 1;
                tot_valid_image = tot_valid_image + 1;
                im_full_path = sprintf('%s%s', im_subfolder, im_name_str);
                im_temp = imread(im_full_path);
                im_subject(:,:,n_valid_image) = im_temp;
                bad_flag = strfind(im_name_str, 'bad');
                if isempty(bad_flag)
                    % Good image
                    im_labels(tot_valid_image) = i;
                else
                    % Bad image
                    im_labels(tot_valid_image) = 39;
                end
            else
                % This is the ambient image that will be substracted from
                % all other images
                im_full_path = sprintf('%s%s', im_subfolder, im_name_str);
                im_ambient = double(imresize(imread(im_full_path), ...
                    [192 168]));
            end
        end
    end
    if n_im_per_person ~= n_valid_image
        warning('Number of valid images mismatch!');
    end
    % Subtract the ambient image from all other images, resize them to the
    % required len/wid level, then normalize with respect to their 2-norm
    for j_image = 1:n_im_per_person
        im_temp = im_subject(:,:,j_image) - im_ambient;
        im_temp = imresize(im_temp, [len wid]);
        im_temp = im_temp(:)/norm(im_temp(:));
        im(:,i*n_im_per_person + j_image) = im_temp;
    end
end
        