function [ ce ] = compCE( T, P )
% Compute classification error,
% T = true labels
% P = predicted labels

% CE = (5 * |FP| + |FN|) / m
ce = sum(((T==-1) & (P==1)) * 5) + sum((T==1) & (P==-1));
ce = ce / length(T);

end