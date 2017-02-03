% Uses advanced library function to compute gradient descent for a classification machine learning problem.
% Requires a specific cost function as an input, which needs to be defined for the problem at hand.

function[optTheta, functionVal, exitFlag] = logisticRegressionGradientDescentAdvanced(costFunction, n)
  
  options = optimset('GradObj', 'on', 'MaxIter', '100');
  initialTheta = zeros(n, 1);
  [optTheta, functionVal, exitFlag] = fminunc(costFunction, initialTheta, options);

end