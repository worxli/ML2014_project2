clear;

%% read data
training = csvread('training.csv');
testdata = csvread('testing.csv');
validation = csvread('validation.csv');

%% 



%% test on validation set


%% write to csv file for submission
csvwrite('validationsetresult.csv', preddata);


%% test on test set


%% write to csv file for submission
csvwrite('testsetresult.csv', preddata);