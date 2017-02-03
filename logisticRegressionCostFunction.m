% Function to compute the cost function and the gradient for logistic regression classification problems

function [J, grad] = classificationCostFunction(theta, X, y)
  
  m = length(y);
  J = 0;
  
  function g = sigmoid(z),
    g = 1 ./ (1 + e.^(-z));
  end
  
  grad = zeros(size(theta));
  h = X * theta;
  h = sigmoid(h);
  J = (1/m) * ((-(y') * log(h)) - (1-y)' * log(1-h));
  
  grad = (1 / m) .* (X' * (h - y'));
  
end