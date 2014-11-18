function [ ce ] = kFoldCV( X, Y, kernelFunc, costM )
% k fold cross validation to train a classifier with a given kernel
% return classifier error

kfold = 10;
ind = crossvalind('Kfold', size(Y,1), kfold);
ce = 0;

for i = 1:kfold
    
    % Select indices for k fold cross validation
    Xts = X(ind == i, 1:end-1);
    Xtr = X(ind ~= i, 1:end-1);

    Yts = Y(ind == i, end);
    Ytr = Y(ind ~= i, end);

    % train svm with the specified kernel function
    svm = fitcsvm(Xtr,Ytr,'Standardize',true,'KernelFunction',kernelFunc,'Cost',costM);
    
    % predict not used training data
    [labels] = predict(svm, Xts);
    ce = ce + compCE(Yts, labels); % accummulate errors
    
end

% mean classification error
ce = ce/kfold;

end