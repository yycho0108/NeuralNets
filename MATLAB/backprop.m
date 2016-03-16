% %% UTILITY
% 
% sigmoid = @(x) 1./(1+exp(-x));
% sigmoidPrime = @(x) sigmoid(x).*(1 - sigmoid(x));
% 
% %% INITIALIZE
% top = [2 4 3 1];
% len = length(top);
% Z = cell(1,len); % "Input"
% A = cell(1,len); % "Activated"
% G = cell(1,len); % "Gradient"
% W = cell(1,len-1);
% 
% for i = 1:len
% 	Z{i} = zeros(top(i),1);%or randn?
% 	A{i} = zeros(top(i),1);
% 	G{i} = zeros(top(i),1);
% end
% 
% for i = 1 : len-1
% 	W{i} = randn(top(i+1),top(i));%zeros(top(i),top(i+1));
% end
% 
% %% FEED-FORWARD
% [I,O] = GEN();
% 
% A{1} = I; %Set Output of Input Array
% for i = 2:length(top)
% 	Z{i} = W{i-1}*A{i-1};
% 	A{i} = sigmoid(Z{i});
% end
% %% BACK-PROPAGATE
% err = 0.5 * sum((O - A{end}).^2);
% G{end} = (O - A{end}) * sigmoidPrime(A{end});
% for i = len-1:-1:2
% 	G{i} = (W{i}' * G{i+1}) .* sigmoidPrime(Z{i});
% end
% 
% for i = 2 : len
% 	dW = G{i} * A{i-1}';
% 	W{i-1} = W{i-1} + 0.6 * dW;
% end
%% TESTING

% top = [2 4 1];
% n = Net(top);
% [I,O] = GEN();
% disp('--');
% n.feedForward(I)
% n.backPropagate(I,O);
% n.feedForward(I)

%% ITERATING
range = 10000; %number of iterations
eta = 0.6; % learning rate

top = [2 4 1];
net = Net(top,eta);
E = zeros(1,range);
for i = 1 : range
	[I,O] = GEN();
	E(i) = net.backPropagate(I,O);
end
plot(E);

%% TESTING
disp(' --- testing --- ');
[I,O] = GEN()
net.feedForward(I)
disp(' --------------- ');