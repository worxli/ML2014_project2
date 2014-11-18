clear;

%% read data
training = csvread('data/training.csv');
testdata = csvread('data/testing.csv');
validation = csvread('data/validation.csv');


%% define data
Xt = training(:,1:end-1);
Y = training(:,end);


%% Define kernel functions and other parameters for svm
% standard matlab svm
% standardize
% define kernel function
% pass cost matrix
% cross validation on/off
kernels = 'rbf';
costM = [0,5;1,0];


%% for all kernel functions do cross validation,
% choose the one with lowest classification error
% ATTENTION: maybe it's better to use the built in crossval(svm) function - (I think it doesn't matter)
ce = kFoldCV(Xt, Y, kernels, costM);

disp('mean weighted classification error:');
disp(ce);

% choose best kernel
bestKernel = 'rbf';

% train the svm with the best kernel function on all training data
svm = fitcsvm(Xt,Y,'Standardize',true,'KernelFunction',bestKernel,'Cost',costM);

% cross validate and classifcation error
cv = crossval(svm);
ce = kfoldLoss(cv,'lossfun','classiferror');

disp('mean classification error:');
disp(ce);


%% predict/test on validation set
[labels,Score] = predict(svm,validation);

% write to csv file for submission
csvwrite('data/validationsetresult.csv', labels);


%% predict/test on test set
%[labels] = predict(svm,validation);

% write to csv file for submission
%csvwrite('data/testsetresult.csv', labels);