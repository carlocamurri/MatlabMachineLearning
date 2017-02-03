% Function to apply feature scaling and mean normalization to a matrix of features X.
% Note: add the column of 1's only AFTER having applied featureNormalize to the feature matrix (you don't want to scale the 1's column).

function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));

  mu = mean(X);
  sigma = std(X);

  for i = 1:size(X, 2),
    X_norm(:,i) = (X_norm(:,i) - mu(i)) / sigma(i);
  end
end