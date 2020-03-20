function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%% Calculate the cost function

% Arrange weight vectors into a weight matrix
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Number of samples
m = size(X, 1);

% Value
J = 0;

% Output value is converted to binary value 0 1
y_label=zeros(m,num_labels);
for i=1:m
    y_label(i,y(i))=1;
end

% Forward propagation
a1=[ones(m,1) X];
z2=Theta1*a1';
a2=sigmoid(z2);
a2=[ones(1,size(a2,2));a2];
z3=Theta2*a2;
a3=sigmoid(z3');

% Cost function without regular term
for i=1:m
    J=J+(-y_label(i,:)*log(a3(i,:)')-(1-y_label(i,:))*log(1-a3(i,:)'))/m;
end

% Cost function with regular term
J=J+lambda/(2*m)*(sum(sum(Theta1(:,2:(input_layer_size + 1)).^2))+sum(sum(Theta2(:,2:(hidden_layer_size + 1)).^2)));

% Back propagation
DELTA1=zeros(hidden_layer_size, input_layer_size+1);
DELTA2=zeros(num_labels, hidden_layer_size+1);
for t=1:m
    delta3=(a3(t,:)-y_label(t,:))';
    delta2=Theta2' * delta3 .* sigmoidGradient([1;z2(:,t)]);
    
    DELTA1=DELTA1 + delta2(2:end)*a1(t,:);
    DELTA2=DELTA2 + delta3 * a2(:,t)';
end

Theta1_grad = DELTA1/m;
Theta2_grad = DELTA2/m;

% Backpropagation adding regular terms
Theta1_R=[zeros(hidden_layer_size,1)  lambda/m * Theta1(:,2:end)];
Theta2_R=[zeros(num_labels,1)  lambda/m * Theta2(:,2:end)];
Theta1_grad = Theta1_grad + Theta1_R;
Theta2_grad = Theta2_grad + Theta2_R;    

% Weight matrix expanded into weight vector
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
