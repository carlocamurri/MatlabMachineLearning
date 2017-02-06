% Function to compute the regularized cost function for a regression problem given a 
% set of parameters theta, a feature matrix X and output values y.
% The parameter lambda is the regularization parameter (>0).

function [J, grad] = linearRegressionRegularizedCostFunction(theta, X, y, lambda)
    
    cost = X * theta - y;
    J = (1 / (2 * m)) * sum((cost) .^ 2, 1) + (lambda / (2 * m)) * sum(theta(2:end) .^ 2, 1);
    
    grad = (1 / m) * sum((X .* cost)', 2) + (lambda / m) * ([0; theta(2:end)]); 
    
end