% Function to compute the regularized cost function for a regression problem given a 
% set of parameters theta, a feature matrix X and output values y.
% The parameter lambda is the regularization parameter (>0).
% Note: n in this example is actually n+1, but was labelled n for simplicity (in Octave, indexes start from 1, so theta(1) is actually 
% theta_0 in common notation).

function cost = regularizedCostFunctionRegression(theta, X, y, lambda)
  cost = 0;
  m = length(y);
  n = length(theta);
  sumSquaredError = sum(((X*theta - y).^2),1);
  thetaToSum = theta(2:n);
  thetaToSum = thetaToSum.^2;
  sumReg = lambda * sum(thetaToSum);
  cost = 1/(2*m) * (sumSquaredError + sumReg);
endfunction