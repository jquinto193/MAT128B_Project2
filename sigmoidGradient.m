function g = sigmoidGradient(z)
%% Calculate the gradient value of the Sigmoid function

g=sigmoid(z).*(1-sigmoid(z));


end
