function dictionary_assignments = optimize_assignments(dictionary, feature_descriptors, lambda)
% OPTIMIZE_ASSIGNMENTS - given a dictionary and a set of feature
% descriptors, finds the responses of the dictionary to the descriptors.
% Used to train the dictionary and when classifying images
    [m, n] = size(feature_descriptors);
    [~, dict_size] = size(dictionary);
    dictionary_assignments = zeros(dict_size, n);
    for i = 1:m
        [u_lasso, stats] = lasso(dictionary, feature_descriptors(:, i), 'Lambda', lambda);
        %u_fs = l1ls_featuresign(double(dictionary'), feature_descriptors(i, :)', lambda);
        
        dictionary_assignments(:, i) = u_lasso;
    end
end

