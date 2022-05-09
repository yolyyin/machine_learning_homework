function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%J
theta0=theta(1);
thetaNot0=theta(2:end);
J=sum((X*theta-y).^2)*1/(2*m)+lambda/(2*m)*sum(thetaNot0.^2);

%grad
grad0=1/m*sum((X*theta-y).*1);
Xfrom2=X(:,2:end);
gradNot0=1/m*Xfrom2'*(X*theta-y)+lambda/m*thetaNot0;
grad=[grad0;gradNot0];





% =========================================================================

grad = grad(:);

end
