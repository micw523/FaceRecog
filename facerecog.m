% Facial Recognition by sparse representation.
addpath ./minFunc_2012/minFunc
addpath ./minFunc_2012/minFunc/compiled
facebase = 'yaleB';
mat_name = sprintf('faces_%s.mat',facebase);
if exist(mat_name,'file')
    load (mat_name);
end
% 30 dictionary elements, 100 iterations, 15 max. nonzero elements
% per image maximum
num_dict = 90*strcmp(facebase,'yaleB') + 30;
num_iter = 50;
max_nnz = 15;
corruption_flag = 0;
occlusion_flag = 0.0; % Occlusion percentage
% 15 subjects, 11 images per subject
num_subject = 38*strcmp(facebase,'yaleB')+15*strcmp(facebase,'yale');
num_im_per_sub = 64*strcmp(facebase,'yaleB')+11*strcmp(facebase,'yale');

%% Load Yale Face Database, downsampled to 60-by-80 by default
options = struct;
options.len = 60 + 36*strcmp(facebase,'yaleB');
options.wid = 80 + 4*strcmp(facebase,'yaleB');
if (~exist('im','var')) && strcmp(facebase,'yale')
    [im, im_label] = faceload(options);
elseif (~exist('im','var')) && strcmp(facebase,'yaleB')
    [im, im_label] = loadYaleB(options);
end
im = im(:,im_label<num_subject+1);
im_label = im_label(im_label<num_subject+1);
num_images = size(im,2);
im_identify = zeros(size(im,1),num_subject);
for i = 1:num_subject
    im_identify(:,i)=im(:,(i-1)*num_im_per_sub+1);
end

%% Check for corruption and occlusion flags
if occlusion_flag > 0
    %load mandrill;
    %[x_len, x_wid] = size(X);
    occ_area = occlusion_flag * options.len * options.wid;
    occ_len = floor(sqrt(occ_area / options.wid * options.len));
    occ_wid = floor(occ_len * options.wid / options.len);
    for i = 1:num_images
        im_r = reshape(im(:,i),[options.len options.wid]);
        X = imresize(im_r, [occ_len occ_wid]);
        % X = X / norm(X(:));
        [x_len, x_wid] = size(X);
        x_pos_occ = randi([1 options.len-x_len+1]);
        y_pos_occ = randi([1 options.wid-x_wid+1]);
        im_r(x_pos_occ:x_pos_occ+x_len-1, y_pos_occ:y_pos_occ+x_wid-1)=...
            X;
        im(:,i) = im_r(:);
    end
end
%% Randomly permute the images
% Set seed to 0 so that results will be the same each run
% rng(0);
rng('shuffle')
I = randperm(num_images);
im = im(:,I);
im_label = im_label(I);
% Separate images into training and testing
cvp = cvpartition(length(im_label),'HoldOut',0.5);
im_train = im(:,cvp.training);
im_test = im(:,cvp.test);
im_label_train = im_label(cvp.training);
im_label_test = im_label(cvp.test);

%% Train a set of images
[dic_mtx_new, sparse_X_new] = k_svd(im_train, num_dict, num_iter, max_nnz);
F_new = norm(dic_mtx_new*sparse_X_new-im_train,'fro')/size(im_train,2);
if exist('F','var')
    if F_new < F
        F = F_new;
        dic_mtx = dic_mtx_new;
        sparse_X = coeff_solve(im_train, dic_mtx, max_nnz);
        save(mat_name,'im','im_label','dic_mtx','sparse_X','F');
    else
        sparse_X = coeff_solve(im_train, dic_mtx, max_nnz);
    end
else
    F = F_new;
    dic_mtx = dic_mtx_new;
    sparse_X = coeff_solve(im_train, dic_mtx, max_nnz);
    save(mat_name,'im','im_label','dic_mtx','sparse_X','F');
end

%% Softmax regression training
train.X = full(sparse_X); train.y = im_label_train;
% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X];
num_classes = num_subject + 1;
rng('shuffle');
lambda = 0.05;
options = struct('MaxIter', 200, 'progTol', 1e-6);
n = size(dic_mtx,2)+1;
theta = randn(n,num_classes-1)*0.1;
theta(:)=minFunc(@softmax_regression_vec, theta(:), options, train.X,...
    train.y, lambda);
accuracy = multi_classifier_accuracy(theta,train.X,train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);
theta=[theta, zeros(n,1)];

%% Process test set
sp_X_test = coeff_solve(im_test, dic_mtx, max_nnz);
test.X = full(sp_X_test); test.y = im_label_test;
test.X = [ones(1,size(test.X,2)); test.X];
accuracy = multi_classifier_accuracy(theta,test.X,test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);

%% SVM Classifier
[test_acc, train_acc, model] = svm_classifier_simple(train.X(2:end,:)',train.y',...
    test.X(2:end,:)',test.y');
fprintf('Training accuracy: %2.1f%%\n', 100*train_acc);
fprintf('Testing accuracy: %2.1f%%\n', 100*test_acc);