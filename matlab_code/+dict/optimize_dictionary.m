function dictionary = optimize_dictionary(dictionary, dictionary_assignments, feature_descriptors)
% Optimize the dictionary using lagrange dual
    dictionary = dict.lagrange_dual(feature_descriptors, dictionary_assignments, 1, dictionary);
end
