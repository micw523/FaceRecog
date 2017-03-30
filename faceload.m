function [im, im_labels] = faceload(options)
% Batch read script for Yale face database
condition_str = {'centerlight','glasses','happy','leftlight','noglasses',...
    'normal','rightlight','sad','sleepy','surprised','wink'};
if isfield(options, 'len'), len = options.len;
else len = 60; end;
if isfield(options, 'wid'), wid = options.wid;
else wid = 80; end;
im = zeros(len*wid,length(condition_str)*15);
im_labels = zeros(1,length(condition_str)*15);
for j = 1:15
    for i = 1:length(condition_str)
        f_name = sprintf('../yalefaces/subject%02d.%s',j,condition_str{i});
        im_temp = imread(f_name);
        im_temp = imresize(im_temp, [len wid]);
        im_temp = double(im_temp(:))/255;
        im_temp = im_temp - mean(im_temp);
        im(:,(j-1)*length(condition_str)+i) = im_temp/norm(im_temp,2);
        im_labels((j-1)*length(condition_str)+i)=j;
    end
end
