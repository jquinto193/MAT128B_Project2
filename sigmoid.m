function g = sigmoid(z)
% Sigmod function

g = 1.0 ./ (1.0 + exp(-z));
end
