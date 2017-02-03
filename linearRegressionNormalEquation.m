% Function to compute the normal equation for a matrix of features X and a vector of target variables y
% Note: the first column in X should include values of 1

function [theta] = linearRegressionNormalEquation(X, y)

  theta = pinv(X'*X)*X'*y;

end