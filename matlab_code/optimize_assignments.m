function dictionary_assignments = optimize_assignments(dictionary, feature_descriptors, lambda)
% OPTIMIZE_ASSIGNMENTS - given a dictionary and a set of feature
% descriptors, finds the responses of the dictionary to the descriptors.
% Used to train the dictionary and when classifying images
    [~, n] = size(feature_descriptors);
    [~, dict_size] = size(dictionary);
    dictionary_assignments = zeros(dict_size, n);
    for i = 1:n
        [u, stats] = lasso(dictionary, feature_descriptors(:, i), 'Lambda', 0.026);
        %u = l1ls_featuresign(double(dictionary), feature_descriptors(:, i), 0.3);
        
        dictionary_assignments(:, i) = u;
    end
end

