
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Basic Fun
%fprintf('Getting a magic matrix ... \n');
%mama(5)


fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); 
y = data(:, 2);

m = length(y)
X = [ ones(m,1) X ];

[m, n] = size(X);

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);
hold on; % keep previous plot visible


theta_init = zeros(n, 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
  fminunc(@(t)(costFunction(X, y, t)), theta_init, options)


% Plot the input values X against the predicted values X*theta = vector
% plot() automatically connects all calculated points
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure




