function dictionary = optimize_dict(dictionary, dictionary_assignments, feature_descriptors)
% Optimize the dictionary using lagrange dual
    dictionary = lagrange_dual(feature_descriptors, dictionary_assignments, 1, dictionary);
    %dictionary = l2ls_learn_basis_dual(feature_descriptors, dictionary_assignments, 1, dictionary);
end
