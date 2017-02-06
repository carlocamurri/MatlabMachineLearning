% Function to calculate the regularized cost function value and its gradient 
% for a classification problem using logistic regression
% with regularization parameter lambda.

function [J, grad] = logisticRegressionRegularizedCostFunction(theta, X, y, lambda)

  m = length(y);
  n = length(theta);
  
  h = sigmoid(X * theta);
  
  J = (1/m) * ((-(y') * log(h)) - (1-y)' * log(1-h)) + ((theta(2:n))' * ones(n-1, 1));
  J = J + ((lambda ./ (2*m)) .* ((theta(2:n, 1) .^ 2)' * ones(n-1, 1)));
  
  grad = (1/m) .* (X' * (h - y)) + (lambda/m) .* ([0; theta(2:n)]);
end
