% test lasso function
X = randn(100,20); % random subsample of feature descriptors 
% 100: dimension of feature space
% 20: number of feature descriptors
V = randn(100,5); % randomly initialized dictionary
% 100: dimension of feature space
% 5: number of "clusters" ?

% U: learns/represents cluster assignment
U = zeros(5, 20);
for i = 1:20
    U(:, i) = lasso(V, X(:, i), 'Lambda', 0.2);
end

size(U)

% test least-squares solution
V = X * pinv(U);
V = V ./ repmat(sqrt(sum(V.^2)), [size(V, 1), 1]);
