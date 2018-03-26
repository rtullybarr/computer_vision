function U = feature_sign_search(V, X, lambda)
% FEATURE_SIGN_SEARCH

% minimize_x f(x) = || x - Vu || + lambda*|| u ||

    for k = 1:size(X, 1)
        x = X(:, k);
        active_set = sparse(zeros(size(X, 1)));
        signs = zeros(size(X, 1));
        u = sparse(zeros(size(X, 1)));

        % from zero coefficients of x,
        zero_locs = find(~u);
        % select the one which maximizes the partial derivative
        grad = V'*V * sparse(u) - V'*y;
        [maximum, max_index] = max(abs(grad(zero_locs)));
        u_index = zero_locs(max_index);
        
        if maximum > lambda
            active_set(u_index) = 1; % add u_i to active set
            signs(u_index) = sign(u(u_index));
        end
        
        % feature-sign step:
        % Let V_hat be a submatrix of V that only includes columns belonging
        % to the active set:
        V_hat = V(active_set==1, :);
        % u_hat and sign_hat are subvectors of u and sign corresponding to
        % the active set.
        u_hat = u(active_set==1);
        signs_hat = sign(active_set==1);
        
        % minimization step:
        u_new = (V_hat' * V_hat) \ (V_hat' * x - lambda .* signs_hat ./ 2);
        
        % discrete line search from u_hat to u_new:
        % check objective value at u_new and points where coefficients
        % change sign.
        
        if (sign(u_new) == sign(u_hat))
            % optimality 1 achieved
            u(active_set == 1) = u_new;
            line_search = 1;
        else
            steps = (0 - u_hat)./(u_new - u_hat);
            a = 0.5*sum((V(:, (active_set==1))*(u_new - u_hat)).^2);
            Vtx = (V' * x);
            b = (u_hat'*(V_hat' * V_hat)*(u_new - u_hat) - (u_new - u_hat)' * Vtx(active_set==1));
            penalty = lambda*sum(abs(x_hat));
            
            [sorted_steps, ix_lsearch] = sort([steps',1]);
            
            remove_idx=[];
            for i = 1:length(sorted_steps)
                step = sorted_steps(i);
                if step <= 0 || step > 1
                    continue;
                end
                
                u_temp = u_hat + (u_new - u_hat) .* step;
                objective = a*step^2 + b*step + lambda*sum(abs(u_temp));
                if objective < penalty
                    penalty = objective;
                    line_search = step;
                    if step < 1
                        remove_idx = [remove_idx ix_lsearch(i)];
                    end
                elseif objective > penalty
                    break;
                else
                    if (sum(u_hat==0)) == 0
                        line_search = step;
                        penalty = objective;
                        if step < 1
                            remove_idx = [remove_idx ix_lsearch(i)];
                        end
                    end
                end
            end
        end
        
        if line_search > 0
            % update u
            u_new = u_hat + (u_new - u_hat) .* line_search;
            u(active_set==1) = u_new;
            signs(active_set==1) = sign(u_new);
        end

        % if u encounters zero along the line search, then remove it from
        % active set
        if lsearch < 1 && lsearch > 0
            remove_idx = find(abs(u(active_set==1)) < eps);
            u(active_set(remove_idx)) = 0;

            signs(active_set(remove_idx)) = 0;
            active_set(remove_idx) = 0;
        end
        
        % check optimality conditions:
        nonzero_u
    end
end

