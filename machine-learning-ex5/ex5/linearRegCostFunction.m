function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% --- Cost function --- %

% Factors for Cost function
h = X * theta;
error = h - y;
error_sqr = error.^2;

% Regularization term
theta_sqr = theta.^2;
reg_term_cost = lambda / (2*m) * sum(theta_sqr(2:end));    %do not regularize theta(1)

% Cost function with regularization
J = 1/(2*m) * sum(error_sqr) + reg_term_cost;



% --- Gradient --- %

% Gradient for theta_0 without regularization

grad(1) = 1/m * sum( error' * X(:,1));  % theta_0 is without regularization

for (j = 2:length(theta))
  reg_term_grad = (lambda / m * theta(j));
  grad(j) = 1/m * sum( error' * X(:,j)) + reg_term_grad;
end





% =========================================================================

grad = grad(:);

end
