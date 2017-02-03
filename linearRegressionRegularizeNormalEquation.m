% Function to use the normal equation method to solve a regression problem with feature set X and predictions y.
% Regularization is implemented by rescaling theta by using a parameter lambda, in order to avoid overfitting.

function theta = linearRegressionRegularizeNormalEquation(X, y, lambda)
  [~, n] = size(X);
  L = eye(n - 1);
  L = [zeros(n - 1, 1), L];
  L = [zeros(1, n); L];
  theta = pinv(X' * X + lambda * L) * X' * y;
end