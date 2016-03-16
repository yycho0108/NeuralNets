classdef Net < handle
	properties
		ETA = 0.5; %learning rate
		len = 0;
		Z = {};
		A = {};
		G = {};
		W = {};
	end
	methods(Static)
		function res = sigmoid(x)
			res = 1.0 ./(1.0+exp(-x));
		end
		function res = sigmoidPrime(x)
			s = Net.sigmoid(x);
			res = s .* (1.0 - s);
		end
	end
	methods
		function obj = Net(top,eta) % top = topology
			obj.len = length(top);
			obj.Z = cell(1,obj.len); % "Input"
			obj.A = cell(1,obj.len); % "Activated"
			obj.G = cell(1,obj.len); % "Gradient"
			obj.W = cell(1,obj.len-1);
			
			for i = 1:obj.len
				obj.Z{i} = zeros(top(i),1);
				obj.A{i} = zeros(top(i),1);
				obj.G{i} = zeros(top(i),1);
			end

			for i = 1 : obj.len-1
				obj.W{i} = randn(top(i+1),top(i));%zeros(top(i),top(i+1));
			end
			
			if nargin>1
				obj.ETA = eta;
			end
			
		end
		function O = feedForward(obj,I)
			obj.A{1} = I;
			for i = 2:obj.len
				obj.Z{i} = obj.W{i-1}*obj.A{i-1};
				obj.A{i} = Net.sigmoid(obj.Z{i});
			end
			O = obj.A{end};
		end
		function cost = backPropagate(obj,I,O)
			Y = obj.feedForward(I);
			E = O - Y;
			obj.G{end} = E * Net.sigmoidPrime(obj.A{end});
			for i = obj.len-1:-1:2
				obj.G{i} = (obj.W{i}' * obj.G{i+1}) .* Net.sigmoidPrime(obj.Z{i});
			end
			for i = 2 : obj.len
				dW = obj.G{i} * obj.A{i-1}';
				obj.W{i-1} = obj.W{i-1} + 0.2 * dW;
			end
			cost = 0.5 * sum(E.^2);
		end
	end
end