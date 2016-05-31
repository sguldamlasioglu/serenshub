function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

% We calculate J    
hypothesis = sigmoid(X*theta); 
costFun = (-y.*log(hypothesis) - (1-y).*log(1-hypothesis));    
J = (1/m) * sum(costFun) + (lambda/(2*m))*sum(theta(2:length(theta)).^2);

% We calculate grad using the partial derivatives
beta = (hypothesis-y); 
grad = (1/m)*(X'*beta);
temp = theta;  
temp(1) = 0;   % because we don't add anything for j = 0  
grad = grad + (lambda/m)*temp; 
grad = grad(:);

end