clear;

%% read data
training = csvread('data/training.csv');
testdata = csvread('data/testing.csv');
validation = csvread('data/validation.csv');

kfold = 10;

%% 
Xt = training;

ind = crossvalind('Kfold', size(Xt,1), kfold);
err = 0;

% do kfold crossvalidation
for i = 1:kfold
    Xts = Xt(ind == i, 1:end-1);
    Xtr = Xt(ind ~= i, 1:end-1);

    Gts = Xt(ind == i, end);
    Gtr = Xt(ind ~= i, end);
    
    %SVMStruct = svmtrain(Xtr,Gtr);
    %SVMModel = fitcsvm(Xtr,Gtr,'KernelFunction','rbf','Standardize',true,'ClassNames',{'negClass','posClass'});
    
    curerr = 0;
    
    err = err + curerr;
end



return
%% test on validation set

preddata = 0;

%% write to csv file for submission
csvwrite('validationsetresult.csv', preddata);


%% test on test set


%% write to csv file for submission
csvwrite('testsetresult.csv', preddata);