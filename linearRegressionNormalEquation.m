% Function to compute the normal equation for a matrix of features X and a vector of target variables y
% Note: the first column in X should include values of 1

function [theta] = normalEqn(X, y)

  theta = zeros(size(X, 2), 1);

  theta = pinv(X'*X)*X'*y

end