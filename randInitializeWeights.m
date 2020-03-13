function W = randInitializeWeights(L_in, L_out)
%% Random initialization neural network weight matrix

epsilon_init=0.12;
W=rand(L_out , 1 + L_in) * 2 * epsilon_init - epsilon_init;


end
