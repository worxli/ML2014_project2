function [ ce ] = compCE( T, P )
% Compute classification error,
% T = true labels
% P = predicted labels

ce = sum(((T==-1) & (P==1)) * 5) + sum((T==1) & (P==-1));

% CE = (5 * |FP| + |FN|) / m
ce = ce / length(T);

end

