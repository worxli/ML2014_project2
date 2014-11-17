clear;

%% read data
training = csvread('data/training.csv');
testdata = csvread('data/testing.csv');
validation = csvread('data/validation.csv');

kfold = 10;

%% 
Xt = training;

% standard matlab svm
% standardize
% define kernerl function
% pass cost matrix
% cross validation on/off

costM = [0,5;1,0];
svm = fitcsvm(Xt(:,1:end-1), Xt(:,end),'Standardize',true,'KernelFunction','rbf','Cost',costM);

% classification loss for observations not used for training (out of
% sample classification)

cv = crossval(svm);
kfoldLoss(cv)

% Predict training data

[labels,Score] = predict(svm,Xt(:,1:end-1));
CE = compCE(Xt(:,end),labels)

% Predict validation data

[labels,Score] = predict(svm,validation);

% do kfold crossvalidation

ind = crossvalind('Kfold', size(Xt,1), kfold);
err = 0;
% for i = 1:kfold
%     Xts = Xt(ind == i, 1:end-1);
%     Xtr = Xt(ind ~= i, 1:end-1);
% 
%     Gts = Xt(ind == i, end);
%     Gtr = Xt(ind ~= i, end);
% 
%     %use some kernel ...
%     % and train svm
%     
%     
%     %test svm
%     curerr = 0;
%     
%     err = err + curerr;
% end


%% test on validation set


%% write to csv file for submission
csvwrite('data/validationsetresult.csv', labels);


%% test on test set


%% write to csv file for submission
%csvwrite('testsetresult.csv', preddata);