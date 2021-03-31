function [weights] = logistic_train(data, labels, epsilon, maxiter,lr)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
% OUTPUT:
%    weights = (d+1) * 1 vector of weights where the weights correspond to
%              the columns of "data"
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
N= size(data,1);
d= size(data,2)-1;
if ~exist('epsilon', 'var')
    epsilon = 10^-5;
end

if ~exist('maxiter', 'var')
    maxiter = 1000;
end

if ~exist('lr', 'var')
    lr = 10^-2;
end

weights= zeros(d+1,1);

for iteration=1:maxiter
        yhat=predict(data,weights);
        weights=weights-lr.*(data'*(yhat-labels))./N;
        if(sum(abs(yhat-predict(data,weights)))<epsilon)
            break
        end
end





