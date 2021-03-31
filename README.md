# Logistic-Regression

 function  logistic train.m that takes an input data set, a set of binary training labels, and an optional argument that specifies the convergence criterion, and returns a set of logistic weights. Specifically the function have the following form:
 
 function [weights] = logistic train(data, labels, epsilon, maxiter)
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
%
%
%
%
% OUTPUT:
%    weights = (d+1) * 1 vector of weights where the weights correspond to
%              the columns of "data"
iterations to execute (useful when debugging in case your
code is not converging correctly!)
(if unspecified can be set to 1000)
%

The classifier is trained using the first-order gradient descent procedure. It uses either +1/0 label encoding. 
algorithm is tested on the Spam Email data set:

https://github.com/jiayuzhou/CSE847/tree/master/data/alzheimers

There are 57 features and 2 class labels. 
A separate test data set consisting of all rows in the file from row 2001 to 4601 inclusive (and corresponding labels) is created. It has 2 data sets, a training data set with 2000 rows (the first 2000 rows of the original file) and a test data set with 2601 rows. Logistic regression classifier is trained on the first n rows of the training data, n = 200; 500; 800, 1000; 1500, 2000.
