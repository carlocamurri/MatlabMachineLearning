% Function to find the gradient of the cost function for a set of parameters theta,
% also applying regularization with parameter lambda.

function grad = linearRegressionRegularizeCostFunctionGradient(theta, X, y, lambda)
  n = length(theta);
  m = length(y);
  nero = X*theta - y;
  delta = nero .* ones(size(X));
  delta = 1/m * sum((delta .* X)', 2);
  grad = zeros(n, 1);
  grad(1) = delta(1);
  grad(2:n) = delta(2:n) + (lambda/m)*theta(2:n);
end