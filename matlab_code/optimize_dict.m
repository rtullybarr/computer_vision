function dictionary = optimize_dict(dictionary, dictionary_assignments, feature_descriptors)
% Optimize the dictionary using lagrange dual
    dictionary = lagrange_dual(dictionary', dictionary_assignments, feature_descriptors, 1);
end
