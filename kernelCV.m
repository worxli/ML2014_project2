function [ osvm ] = kernelCV(Xt, Y, kernel, costM)

    outlierFrac = 0:0.01:0.05;
    errors = zeros(length(outlierFrac));
    c = cvpartition(length(Xt(:,1)),'KFold',10);
    svms = cell(length(outlierFrac));
    st = true;
    
    parfor i = 1:length(outlierFrac)
        
        outFrac = outlierFrac(i);
        z = [];

        % train svm with the specified kernel function
        minfn = @(z)kfoldLoss(fitcsvm(Xt,Y,'CVPartition',c,...
                'KernelFunction',kernel,'BoxConstraint',exp(z(2)),...
                'OutlierFraction',outFrac,'Standardize', st,...
                'KernelScale',exp(z(1))));
            
        opts = optimset('TolX',5e-4,'TolFun',5e-4);
        [searchmin fval] = fminsearch(minfn,randn(2,1),opts);
        
        z = exp(searchmin);
        
        svm = fitcsvm(Xt,Y, 'Standardize', st, 'BoxConstraint', z(2), 'KernelScale',z(1),'KernelFunction',kernel, 'OutlierFraction',outFrac);

        cv = crossval(svm);
        ce = kfoldLoss(cv,'lossfun','classiferror');

        disp('mean classification error:');
        disp(ce);
        
        errors(i) = ce;
        svms{i} = svm;
        
    end
    
    [err, ind] = min(errors);
    
    osvm = svms{ind};

end

