% Facial Recognition by sparse representation.
addpath ./minFunc_2012/minFunc
addpath ./minFunc_2012/minFunc/compiled
if exist('faces.mat','file')
    load faces.mat;
end
% 30 dictionary elements, 100 iterations, 15 max. nonzero elements
% per image maximum
num_dict = 30;
num_iter = 100;
max_nnz = 15;
% 15 subjects, 11 images per subject
num_subject = 15;
num_im_per_sub = 11;

%% Load Yale Face Database, downsampled to 60-by-80 by default
options = struct;
options.len = 60;
options.wid = 80;
if ~exist('im','var')
    [im, im_label] = faceload(options);
end
num_images = size(im,2);
im_identify = zeros(size(im,1),num_subject);
for i = 1:num_subject
    im_identify(:,i)=im(:,(i-1)*num_im_per_sub+1);
end

%% Randomly permute the images
% Set seed to 0 so that results will be the same each run
% rng(0);
rng('shuffle')
I = randperm(num_images);
im = im(:,I);
im_label = im_label(I);
% Separate images into training and testing
im_train = im(:,1:120);
im_test = im(:,121:end);
im_label_train = im_label(1:120);
im_label_test = im_label(121:end);

%% Train a set of images
[dic_mtx_new, sparse_X_new] = k_svd(im_train, num_dict, num_iter, max_nnz);
F_new = norm(dic_mtx_new*sparse_X_new-im_train,'fro');
if exist('F','var')
    if F_new < F
        F = F_new;
        dic_mtx = dic_mtx_new;
        sparse_X = coeff_solve(im_train, dic_mtx, max_nnz);
        save('faces.mat','im','im_label','dic_mtx','sparse_X','F');
    else
        sparse_X = coeff_solve(im_train, dic_mtx, max_nnz);
    end
else
    F = F_new;
    dic_mtx = dic_mtx_new;
    sparse_X = coeff_solve(im_train, dic_mtx, max_nnz);
    save('faces.mat','im','im_label','dic_mtx','sparse_X','F');
end

%% Softmax regression training
train.X = full(sparse_X); train.y = im_label_train;
% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X];
num_classes = num_subject + 1;
rng('shuffle');
lambda = 0.1;
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

