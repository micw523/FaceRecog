function [f,g] = softmax_regression_vec(theta, X, y, lambda)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  for idx = 1:num_classes-1
      f=f-(y==idx)*log(computeProb(theta,idx,X))';
      g(:,idx)=-X*((y==idx)-computeProb(theta,idx,X))';
  end
%   f=f-(y==num_classes)*log(ones(size(y))./(exp(sum(theta'*X,1)+length(y))))';
%   g(:,num_classes)=-X*((y==num_classes)-ones(size(y))./(exp(sum(theta'*X,1)+length(y))))';
% Regularization
  for idx = 2:n
    f=f+lambda/2*sum(theta(idx,:).^2);
    g(idx,:)=g(idx,:)+lambda*theta(idx,:);
  end
  g=g(:); % make gradient a vector for minFunc

    function p=computeProb(theta,idx,X)
        p=exp(theta(:,idx)'*X)./(sum(exp(theta'*X),1));
    