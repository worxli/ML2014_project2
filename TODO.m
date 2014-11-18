
% READ SVM HELP DOCUMENTATION -> TUNING AN SVM CLASSIFIER

% Tuning an SVM Classifier
% Try tuning parameters of your classifier according to this scheme:

% Pass the data to fitcsvm, and set the name-value pair arguments 'KernelScale','auto'. 
% Suppose that the trained SVM model is called SVMModel. The software uses a heuristic 
% procedure to select the kernel scale. The heuristic procedure uses subsampling. Therefore, 
% to reproduce results, set a random number seed using rng before training the classifier.
% Cross validate the classifier by passing it to crossval. By default, the software conducts 
% 10-fold cross validation.
% Pass the cross-validated SVM model to kFoldLoss to estimate and retain the classification error.
% Retrain the SVM classifier, but adjust the 'KernelScale' and 'BoxConstraint' name-value pair arguments.
% BoxConstraint — One strategy is to try a geometric sequence of the box constraint parameter. 
% For example, take 11 values, from 1e-5 to 1e5 by a factor of 10. Increasing BoxConstraint might decrease
% the number of support vectors, but also might increase training time.
% KernelScale — One strategy is to try a geometric sequence of the RBF sigma parameter 
% scaled at the original kernel scale. Do this by:
% Retrieving the original kernel scale, e.g., ks, using dot notation: ks = SVMModel.KernelParameters.Scale.
% Use as new kernel scales factors of the original. For example, multiply ks by 
% the 11 values 1e-5 to 1e5, increasing by a factor of 10.
% Choose the model that yields the lowest classification error.

% You might want to further refine your parameters to obtain better accuracy. 
% Start with your initial parameters and perform another cross-validation step, this time 
% using a factor of 1.2. Alternatively, optimize your parameters with fminsearch, as shown in 
% Train and Cross Validate SVM Classifiers.