% Function to perform gradient descent on a regression machine learning problem. It is used to find optimal values for the vector 'theta',
% which should include n parameters for n features (columns) in the feature set X (including the first column of 1's)
% The algorithm decrements the value of input vector theta by 
% a parameter alpha (usually between 0 and 1) * the derivative of the cost function J(theta), 
% which should converge to an optimal value for theta after a set number of iterations
% Note: featureNormalize should be applied beforehand on the feature set X (even before the column of 1's is added)

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters

      lambda = X*theta - y;
      delta = lambda .* ones(size(X));
      delta = 1/m * sum((delta .* X)', 2);
      
      theta = theta - alpha * delta;
      
      % Storing the value of the cost function for every iteration to check for convergence (and to possibly graph the values)
      J_history(iter) = computeCostMulti(X, y, theta);

  end

end