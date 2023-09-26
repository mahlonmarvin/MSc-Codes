% INITIALIZE THE NEURAL NETWORK PROBLEM %

% inputs for the neural net(AND gate example== 2 inputs && 4 samples)
inputs = LHV';

% targets for the neural net
targets = H2';

% number of neurons
n = 10;

% create a neural network
net = feedforwardnet(n);

% configure the neural network for this dataset
net = configure(net, inputs, targets);

% get the normal NN weights and bias
w1 = getwb(net);

% error MSE normal NN
error1 = targets - net(inputs);
calc = mean(error1.^2)/mean(var(targets',1))

% create handle to the MSE_TEST function, that
% calculates MSE
h = @(x) NMSE(x, net, inputs, targets);

% Setting the Genetic Algorithms tolerance for
% minimum change in fitness function before
% terminating algorithm to 1e-8 and displaying
% each iteration's results.

ga_opts = gaoptimset('PopInitRange', [-1;1], 'TolFun', 1e-10,'display','iter');
ga_opts = gaoptimset(ga_opts, 'StallGenLimit', 100, 'FitnessLimit', 1e-5, 'Generations', 20);

% PLEASE NOTE: For a feed-forward network
% with n hidden neurons, 3n+n+1 quantities are required
% in the weights and biases column vector.
% a. n for the input weights=(features*n)=2*n
% b. n for the input biases=(n bias)=n
% c. n for the output weights=(n weights)=n
% d. 1 for the output bias=(1 bias)=1

% running the genetic algorithm with desired options
[x, err_ga] = ga(h, 2*n+n+n+1, ga_opts);
net = setwb(net, x');

% get the GA optimized NN weights and bias
w2 = getwb(net);

% error MSE GA optimized NN
error2 = targets - net(inputs);
MSE = mean(error2.^2)/mean(var(targets',1))
