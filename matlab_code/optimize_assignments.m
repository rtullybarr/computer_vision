function dictionary_assignments = optimize_assignments(dictionary, feature_descriptors)
% OPTIMIZE_ASSIGNMENTS - given a dictionary and a set of feature
% descriptors, finds the responses of the dictionary to the descriptors.
% Used to train the dictionary and when classifying images
    [m, n] = size(feature_descriptors);
    [dict_size, ~] = size(dictionary);
    dictionary_assignments = zeros(dict_size, n);
    for i = 1:m
        dictionary_assignments(:, i) = lasso(transpose(dictionary), feature_descriptors(i, :), 'Lambda', 0.2);
    end
end

