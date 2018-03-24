function dictionary = lagrange_dual(X, U, l2norm, V)
%LAGRANGE_DUAL 
% learn dictionary bases given coefficients and features
% minimize || feature_descriptors - dictionary * dictionary_assignments ||^2
    
    dual_lambda = diag((V \ (X * U')) - U * U');
    lb=zeros(size(dual_lambda));
    %options = optimoptions('fmincon', 'Algorithm', 'trust-region-reflective', 'SpecifyObjectiveGradient', true, 'HessianFcn', 'objective');
    
    XUt = X * U';
    trXXt = sum(sum(X.^2));
    UUt = U*U';
    
    options = optimset('GradObj', 'on', 'Display', 'off');
    [new_lambdas, ~] = fmincon(@(new_lambdas) basis_objective(new_lambdas, XUt, trXXt, UUt, l2norm^2), dual_lambda, [], [], [], [], lb, [], [], options);
    
    dual_lambda = new_lambdas;
    lambda_diag = diag(dual_lambda);
    
    % assign new dictionary now that we have optimized the lagrange dual
    V = (U*U' + lambda_diag) \ (X*U')';
    dictionary = V';
end

function [f, g, h] = basis_objective(dual_lambda, XUt, trXXt, UUt, c)
    % lagrange dual, D(lambda):
    % trace(X'X?XU' *(UU'+ ?)^-1 * (XU')' ? c?)
    L= size(XUt,1);
    M= length(dual_lambda);

    UUt_inv = inv(UUt + diag(dual_lambda));

    % trXXt = sum(sum(X.^2));
    if L>M
        % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
        f = -trace(UUt_inv*(XUt'*XUt))+trXXt-c*sum(dual_lambda);

    else
        % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
        f = -trace(XUt*UUt_inv*XUt')+trXXt-c*sum(dual_lambda);
    end
    f= -f;

    if nargout > 1   % fun called with two output arguments
        % Gradient of the function evaluated at x
        g = zeros(M,1);
        temp = XUt*UUt_inv;
        g = sum(temp.^2) - c;
        g= -g;
        %gs = sum(g)


        if nargout > 2
            % Hessian evaluated at x
            % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
            H = -2.*((temp'*temp).*UUt_inv);
            H = -H;
            %hs = sum(H)
        end
    end
end

