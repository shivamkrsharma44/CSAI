%
% function to learn a decoder for semantic vectors from imaging data
%
% (derived from code written by Alona Fyshe and Mark Palatucci in
%  Tom Mitchell's group at Carnegie Mellon University)
%
% input
% - trainingData    - a #examples x #voxels matrix
% - trainingTargets - a #examples x #dimensions matrix
%
% output:
% - weighMatrix - a #voxels+1 x #dimensions weight matrix
% - r - #dimensions vector with the regularization parameter value for each dimension
%
% Notes:
% - column i of weightMatrix has #voxels weights + intercept (last row)
%   for predicting target i
% - This function uses an efficient implementation of cross-validation within the
%   training set to pick a different optimal value of the
%   regularization parameter for each semantic dimension in the target vector
% - This function uses kernel ridge regression with a linear
%   kernel. This allows us to use the full brain as input features
%   because we avoid the large inversion of the voxels/voxels matrix

function [weightMatrix, r] = learnDecoder(trainingData, trainingTargets)

% append a vector of ones so we can learn weights and biases together
trainingData(:,end+1) = 1;
numDimensions = size(trainingData,2);
numTargetDimensions = size(trainingTargets,2);
numExamples = size(trainingData,1);

params = [1 .5 5 0.1 10 0.01 100 0.001 1000 0.0001 10000 0.00001 100000 0.000001 1000000]
n_params = length(params);
n_targs = size(trainingTargets,2);
n_words = size(trainingData,1);

CVerr = zeros(n_params, n_targs);

% If we do an eigendecomp first we can quickly compute the inverse for many different values
% of lambda. SVD uses X = UDV' form.
% First compute K0 = (XX' + lambda*I) where lambda = 0.
K0 = trainingData*trainingData';
[U,D,V] = svd(K0);

for i = 1:length(params)
    regularizationParam = params(i);
    %    fprintf('CVLoop: Testing regularation param: %f, ', regularizationParam);
    
    
    % Now we can obtain Kinv for any lambda doing Kinv = V * (D + lambda*I)^-1 U'
    dlambda = D + regularizationParam*eye(size(D));  % this D is not squared because it's computed from XX^T
    dlambdaInv = diag(1 ./ diag(dlambda));
    KlambdaInv = V * dlambdaInv * U';
    
    % Compute pseudoinverse of linear kernel.
    KP = trainingData' * KlambdaInv;
    
    % Compute S matrix of Hastie Trick X*KP
    S = trainingData * KP;
    
    % Solve for weight matrix so we can compute residual
    weightMatrix = KP * trainingTargets;
    
    % Original code for none kernel version
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %weightMatrix = (trainingData'*trainingData + regularizationParam*eye(numDimensions)) \
    %trainingData'* trainingTargets;
    % compute the cross validation error using Hastie CV trick
    %S = trainingData*inv(trainingData'*trainingData + regularizationParam*eye(numDimensions))* trainingData';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    Snorm = repmat(1 - diag(S), 1, numTargetDimensions);
    YdiffMat = (trainingTargets - (trainingData*weightMatrix));
    YdiffMat = YdiffMat ./ Snorm;
    CVerr(i,:) = (1/numExamples).*sum(YdiffMat .* YdiffMat);
end

% find min errs to choose best reg param
[minerr, minerrIndex] = min(CVerr);
r=zeros(1,n_targs);

for cur_targ = 1:n_targs,
    regularizationParam = params(minerrIndex(cur_targ));
    r(cur_targ) = regularizationParam;
    
    % got good param, now obtain weights
    dlambda = D + regularizationParam*eye(size(D));
    dlambdaInv = diag(1 ./ diag(dlambda));
    KlambdaInv = V * dlambdaInv * U';
    
    % Solve for weight matrix so we can compute residual
    weightMatrix(:,cur_targ) = trainingData' * KlambdaInv * trainingTargets(:,cur_targ);
end


