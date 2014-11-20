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
standardize = false;
kernel = 'rbf';
costM = [0,5;1,0];


%% for all kernel functions do cross validation,
svm = kernelCV(Xt, Y, kernel, costM);


%% predict/test on validation set
[labels,Score] = predict(svm,validation);

% write to csv file for submission
csvwrite('data/validationsetresult.csv', labels);


%% predict/test on test set
[labels] = predict(svm,testdata);

% write to csv file for submission
csvwrite('data/testsetresult.csv', labels);