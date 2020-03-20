function p = predict(Theta1, Theta2, X)
%% Neural network for prediction based on given network weights
% Number of samples
m = size(X, 1);
% Number of output categories
num_labels = size(Theta2, 1);

% Calculating neural network output
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[D , P] = max(h2, [], 2);

end

function g = sigmoid(z)
% Sigmod function

g = 1.0 ./ (1.0 + exp(-z));
end

