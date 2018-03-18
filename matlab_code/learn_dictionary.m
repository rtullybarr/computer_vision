function dictionary = learn_dictionary(feature_descriptors, num_clusters, num_iterations)
%LEARN_DICTIONARY sparse codes the given set of feature descriptors
% using the feature-sign / lasso algorithm
    [m, n] = size(feature_descriptors);
    % n: dimension of feature space
    % m: number of feature descriptors
    
    % initialize dictionary: random set of feature descriptors
    perm = randperm(m);
    dictionary = feature_descriptors(perm(1:num_clusters), :);
    %dictionary = transpose(dictionary);

    for i = 1:num_iterations
        % Steps: (Iterate)
        % Fix dictionary and optimize dictionary_assignments
        dictionary_assignments = optimize_assignments(dictionary, feature_descriptors);

        % Fix dictionary_assignments and optimize dictionary
        new_dictionary = optimize_dict(dictionary_assignments, feature_descriptors);

        % return dictionary when optimization doesn't change much
        diff = abs(new_dictionary - dictionary);
        dictionary = new_dictionary;
        diff = sum(diff(:))
        if diff < 0.01
            break;
        end
    end
end

