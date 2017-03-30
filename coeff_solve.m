function sparse_X = coeff_solve (data, dic_mtx, max_nnz)
[~,num_k] = size(dic_mtx);
[~,n_image] = size(data);
sparse_irow = zeros(max_nnz*n_image,1);
sparse_icol = zeros(max_nnz*n_image,1);
sparse_ival = zeros(max_nnz*n_image,1);
tot_elem = 1;
for i=1:n_image
    % Solve for the sparse coefficients for vector x
    x = data(:,i);
    res = x;
    x_pos = zeros(max_nnz,1);
    for j=1:max_nnz
        proj_vec = dic_mtx'*res;
        [~,pos] = max(abs(proj_vec));
        % Direction of max. null space
        pos = pos(1);
        x_pos(j) = pos;
        coeff_i = pinv(dic_mtx(:,x_pos(1:j)))*x;
        res = x - dic_mtx(:,x_pos(1:j))*coeff_i;
        if norm(res) < 1e-4 
            break;
        end
    end
    n_len = length(coeff_i);
    sparse_icol(tot_elem:tot_elem+n_len-1) = i;
    sparse_irow(tot_elem:tot_elem+n_len-1) = x_pos(1:n_len);
    sparse_ival(tot_elem:tot_elem+n_len-1) = coeff_i;
    tot_elem = tot_elem + n_len;
end
sparse_icol = sparse_icol (1:tot_elem-1);
sparse_irow = sparse_irow (1:tot_elem-1);
sparse_ival = sparse_ival (1:tot_elem-1);
sparse_X = sparse(sparse_irow, sparse_icol, sparse_ival, num_k, n_image);
