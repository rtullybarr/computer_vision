function dictionary = sparse_coding(feature_descriptors, dict_size, num_iterations, lambda)
%LEARN_DICTIONARY sparse codes the given set of feature descriptors
% using the feature-sign / lasso algorithm
    [~, n] = size(feature_descriptors);
    % n: dimension of feature space
    % m: number of feature descriptors
    
    % initialize dictionary: random set of feature descriptors
    perm = randperm(n);
    dictionary = feature_descriptors(:, perm(1:dict_size));

    for i = 1:num_iterations
        % Steps: (Iterate)
        % Fix dictionary and optimize weights
        weights = dict.optimize_weights(dictionary, feature_descriptors, lambda);

        % Fix weights and optimize dictionary
        new_dictionary = dict.optimize_dictionary(dictionary, weights, feature_descriptors);

        % return dictionary when optimization doesn't change much
        diff = abs(new_dictionary - dictionary);
        dictionary = new_dictionary;
        diff = sum(diff(:));
        disp(diff)
        if diff < 0.0001
            break;
        end
    end
end

