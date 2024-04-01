function H = bilinearfeatures(X)
%bilinearfeatures 
% Computes the feature vector for a bilinear model
% X = Raw features x_1 ... x_D, arranged as a N by D matrix
[N,D] = size(X);
H =zeros(N, 1 + D + D*(D+1)/2);

% Zeroeth order
H(:,1) = 1;
% First order
H(:,1 + (1:D)) = X;
% Second order
for d = 1:D
    H(:,D+1 + (1:d) + d*(d-1)/2) = X(:,1:d).*X(:,d);
end
end

