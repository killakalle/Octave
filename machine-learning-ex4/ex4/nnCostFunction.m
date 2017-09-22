function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
% The matrices Theta1 and Theta2 will now be in your workspace
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Add ones to the X data matrix
a1 = [ones(m, 1) X];



% --- Forward Propagation --- %

a2 = sigmoid(Theta1 * a1'); 

%Add bias node to a2
a2 = [ones(1, size(a2,2)); a2];

a3 = sigmoid(a2' * Theta2');
hx = a3;



% --- Cost Calculation --- %

% Re-code y into vectors of 0s and 1s
y_matrix = eye(num_labels)(y,:);

% Calculate cost (non-vectorized)
for i=1:m
  for k=1:num_labels
  
    J = J + ( ...
        -y_matrix(i,k) * log(hx(i, k)) ...
          - (1 - y_matrix(i,k)) * log(1 - hx(i, k))  );
  end
end
J = (1/m) * J;

% ALTERNATIVE: Calculate cost (vectorized)
% See the following discussion
% https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA


% Calculate regularization
R1 = 0;
R2 = 0;

[rowsT1 colsT1] = size(Theta1);
[rowsT2 colsT2] = size(Theta2);

% Regularize Theta1
% We are not regularizing terms that correspond to bias unit (= first column of Theta1)
for j=1:rowsT1
  for k=2:colsT1
    R1 = R1 + Theta1(j, k)^2;
  end
end

% Regularize Theta2
% We are not regularizing terms that correspond to bias unit (= first column of Theta2)
for j=1:rowsT2
  for k=2:colsT2
    R2 = R2 + Theta2(j, k)^2;
  end
end


R = (lambda / (2 * m)) * (R1 + R2);


% Add regularization to Cost
J = J + R;


% --- Gradient Calculation - Back Propagation --- %

% This is vectorized backpropagation, thus not using for loops

% Expected dimensions

% a1: 5000x401
% z2: 5000x25
% a2: 5000x26
% a3: 5000x10
% d3: 5000x10
% d2: 5000x25
% Theta1, Delta1 and Theta1_grad: 25x401
% Theta2, Delta2 and Theta2_grad: 10x26
 
% Step 1 - Feed forward
a1 = [ones(rows(X), 1) X];       %add bias unit
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(rows(a2), 1) a2];       %add bias unit
z3 = a2 * Theta2';
a3 = sigmoid(z3);
  
% Step 2
d3 = a3 - y_matrix;

% Step 3

% Exclude first column of Theta2 because it belongs to bias unit
% Bias unit is not considered for backpropagation
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

% Step 4
Delta1 = d2' * a1;
Delta2 = d3' * a2;

%{ 
size_a1 = size(a1); 
size_z2 = size(z2);
size_a2 = size(a2);
size_a3 = size(a3);
size_d3 = size(d3);
size_d2 = size(d2);
size_D1 = size(Delta1);
size_D2 = size(Delta2);


fprintf('a1: 5000x401 vs %dx%d\n', size_a1);
fprintf('z2: 5000x25 vs %dx%d\n', size_z2);
fprintf('a2: 5000x26 vs %dx%d\n', size_a2);
fprintf('a3: 5000x10 vs %dx%d\n', size_a3);
fprintf('d3: 5000x25 vs %dx%d\n', size_d3);
fprintf('d2: 5000x25 vs %dx%d\n', size_d2);
fprintf('D1: 25x401 vs %dx%d\n', size_D1);
fprintf('D2: 10x26 vs %dx%d\n', size_D2);
   
%}  
  
%end

% Step 5 - divide by 1/m
Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;


% Regularize Gradients

for j=1:rowsT1
  for k=2:colsT1
    Theta1_grad(j,k) = Theta1_grad(j,k) + (lambda / m) * Theta1(j,k);
  end
end


for j=1:rowsT2
  for k=2:colsT2
   Theta2_grad(j,k) = Theta2_grad(j,k) + (lambda / m) * Theta2(j,k);
  end
end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
