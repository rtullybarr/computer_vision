function weights = optimize_weights(dictionary, feature_descriptors, lambda)
% OPTIMIZE_ASSIGNMENTS - given a dictionary and a set of feature
% descriptors, finds the responses of the dictionary to the descriptors.
% Used to train the dictionary and when classifying images
    [~, n] = size(feature_descriptors);
    [~, dict_size] = size(dictionary);
    weights = zeros(dict_size, n);
    for i = 1:n
        [u, stats] = lasso(dictionary, feature_descriptors(:, i), 'Lambda', lambda);
        
        weights(:, i) = u;
    end
end

