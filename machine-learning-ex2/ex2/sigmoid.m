function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
dim_z = size(z);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


for i=1:dim_z(1)
  for j=1:dim_z(2)
    g(i,j) = 1 / (1 + exp(-z(i,j)));
   end
end


% =============================================================

end
