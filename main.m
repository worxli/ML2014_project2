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

%standard matlab svm
svm = fitcsvm(Xt(:,1:end-1), Xt(:,end),'Standardize',true);

cv = crossval(svm);
kfoldLoss(cv)

return

% do kfold crossvalidation
for i = 1:kfold
    Xts = Xt(ind == i, 1:end-1);
    Xtr = Xt(ind ~= i, 1:end-1);

    Gts = Xt(ind == i, end);
    Gtr = Xt(ind ~= i, end);

    %use some kernel ...
    % and train svm
    
    
    %test svm
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