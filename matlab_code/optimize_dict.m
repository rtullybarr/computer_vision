function dictionary = optimize_dict(dictionary_assignments, feature_descriptors)
% Optimize the dictionary
    % least-squares solution
    size(dictionary_assignments)
    size(feature_descriptors)
    dictionary = dictionary_assignments * feature_descriptors;
    dictionary = normalize(dictionary); 
    %dictionary = transpose(dictionary);
end
