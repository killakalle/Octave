function [J, grad] = costFunction(X, y, theta)
%COSTFUNCTION Compute ... 
%   [J, grad] = costFunction(X, y, theta) ...

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================

s = ones(1,m);

%size_s = size(s)
%size_X = size(X)
%size_theta = size(theta)

J = 1 / (2*m) * (s * (X * theta - y).^2);    %using s to summarize result vector into single value

% vectorized NOT WORKING PROPERLY
%grad = 1/(2*m) * X' * (X * theta - y);

% Gradient non-vectorized 
        
%{        
for j=1:length(theta)
  for i=1:m
    grad(j) = grad(j) + ...
                1/m * (theta' * X(i,:)' - y(i)) * X(i,j);
  end
end
%}


% Gradient vectorized
grad = (1 / m) * X' * (X * theta - y);




% =========================================================================

end
