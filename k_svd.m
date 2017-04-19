function [dic_mtx, sparse_X] = k_svd (data, num_k, num_iter, max_nnz)
% K-SVD implementation for EECS 556
% input: data: image_len - by - n_image data matrix
%        num_k: number of dictionary elements
%        num_iter: number of training iterations
%        max_nnz: max. number of non-zero elements for one image
% output: dic_mtx: dictionary matrix
%         sparse_X: the sparse coefficients of data using the trained
%                   dictionary matrix
% This specific implementation will initialize the dictionary matrix
% by the data vectors provided.

[image_len, n_image] = size(data);
if (num_k>n_image)
    % No need for training, number of dictionary elements more then
    % number of images provided
    dic_mtx = data;
    sparse_X = eye(n_image);
else
    % Actual training
    % Initialize by the image vectors. Pool the images first
    % The solution will not be guaranteed to be the same for 
    % two different runs
    I = randperm(n_image);
    data = data(:,I);
    dic_mtx = data(:,1:num_k);
    % Normalize the dictionary and keep the first element positive
    for i=1:num_k
        dic_mtx(:,i)=dic_mtx(:,i).*sign(dic_mtx(1,i))./norm(dic_mtx(:,i));
    end
    for i=1:num_iter
        sparse_X = coeff_solve(data, dic_mtx, max_nnz);
        I = randperm(num_k);
        for j = I
            [dict_vec, sparse_X] = dict_train(data, dic_mtx, j, sparse_X);
            dic_mtx(:,j) = dict_vec;
        end
        fprintf('Training Iteration %d Completed\n', i);
    end
end
dic_mtx = clear_dict(dic_mtx, sparse_X, data);

function [dict_vec, sparse_X] = dict_train(data, dic_mtx, j, sparse_X)
dic_j_used = find(sparse_X(j,:));
if isempty(dic_j_used) % No one is using this dictionary
    % Find a better dictionary for this vector
    err = data - dic_mtx*sparse_X;
    err_norm = sum(err.^2);
    [~,i] = max(err_norm);
    dict_vec = data(:,i);
    dict_vec = dict_vec / norm(dict_vec) * sign(dict_vec(1));
else
    % Improve upon existing dictionary
    sub_sparse_X = sparse_X(:, dic_j_used);
    sub_sparse_X(j,:) = 0;
    err = data(:, dic_j_used) - dic_mtx*sub_sparse_X;
    [dict_vec, s, v] = svds(err, 1);
    sparse_X(j,dic_j_used) = s*v';
end

function dic_mtx = clear_dict(dic_mtx,sparse_X,data)
T2 = 0.99;
T1 = 3;
K=size(dic_mtx,2);
Er=sum((data-dic_mtx*sparse_X).^2,1); % remove identical atoms
G=dic_mtx'*dic_mtx; G = G-diag(diag(G));
for jj=1:1:K,
    if max(G(jj,:))>T2 || length(find(abs(sparse_X(jj,:))>1e-7))<=T1 ,
        [~,pos]=max(Er);
        Er(pos(1))=0;
        dic_mtx(:,jj)=data(:,pos(1))/norm(data(:,pos(1)));
        G=dic_mtx'*dic_mtx; G = G-diag(diag(G));
    end;
end;
