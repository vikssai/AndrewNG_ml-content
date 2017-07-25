function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = C_vec;
pred_mat = zeros(size(C_vec)(1),size(sigma_vec)(1));
C_mat = double(zeros(size(C_vec)(1),size(sigma_vec)(1)));
sigma_mat = zeros(size(C_vec)(1),size(sigma_vec)(1));

for i = 1:length(C_vec),
	C = C_vec(i);
	for j = 1:length(sigma_vec),
		sigma = sigma_vec(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		predictions = svmPredict(model, Xval);
		pred_error = mean(double(predictions ~= yval));
		
		pred_mat(i,j) = pred_error;
		C_mat(i,j) = C;
		sigma_mat(i,j) = sigma;
		end;
	end;
end;ex6
[x, ix] = min(min(pred_mat));
C = C_mat(x, ix);
sigma = sigma_mat(x, ix);		


% =========================================================================

end
