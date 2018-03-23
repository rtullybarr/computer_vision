function dictionary = lagrange_dual(X, U, l2norm, V)
%LAGRANGE_DUAL 
% learn dictionary bases given coefficients and features
% minimize || feature_descriptors - dictionary * dictionary_assignments ||^2
    
    dual_lambda = diag((V \ (X * U')) - U * U');
    lb=zeros(size(dual_lambda));
    %options = optimoptions('fmincon', 'Algorithm', 'trust-region-reflective', 'SpecifyObjectiveGradient', true, 'HessianFcn', 'objective');
    options = optimset('GradObj', 'on');
    [new_lambdas, ~] = fmincon(@(new_lambdas) basis_objective(new_lambdas, U, X, l2norm^2), dual_lambda, [], [], [], [], lb, [], [], options);
    
    dual_lambda = new_lambdas;
    lambda_diag = diag(dual_lambda);
    
    % assign new dictionary now that we have optimized the lagrange dual
    V = (U*U' + lambda_diag) \ (X*U')';
    dictionary = V';
end

function [result, gradient, hessian] = basis_objective(dual_lambda, U, X, c)
    % lagrange dual, D(lambda):
    % trace(X'X?XU' *(UU'+ ?)^-1 * (XU')' ? c?)
    
    XUt = X * U';
    L = size(XUt, 1);
    M = length(dual_lambda);
    trXXt = sum(sum(X.^2));
    
    UUt = U*U';
    UUt_inv = inv(UUt + diag(dual_lambda));
    
    if L>M
        % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
        result = trace(UUt_inv*(XUt'*XUt)) - trXXt + c*sum(dual_lambda);

    else
        % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
        result = trace(XUt*UUt_inv*XUt') - trXXt + c*sum(dual_lambda);
    end
    
    % Gradient of D(lambda)
    % ?XU'*inv(U*U' + ?)ei?^2 ? c
    if nargout > 1
        gradient = c - sum(X*U'*UUt_inv).^2;
    end
    
    % Hessian of D(lambda)
    if nargout > 2
        hessian = 2 .* ((X*U'*UUt_inv)'*(X*U'*UUt_inv)).*UUt_inv;
    end
end

