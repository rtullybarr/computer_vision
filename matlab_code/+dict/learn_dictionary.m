function dictionary = learn_dictionary(features, training_set_fraction, dictionary_size, iterations, lambda)
%LEARN DICTIONARY - uses the provided features to learn a dictionary
    
    features_flat = cell(length(features), 1);
    for i = 1:length(features)
        % desc: a RxCxd matrix of descriptors
        desc = features{i};
        [~, ~, d] = size(desc);
        features_flat{i} = reshape(desc, [], d);
        features_flat{i} = features_flat{i}';
    end

    features_all = cat(2, features_flat{:});

    % select a percentage of the descriptors
    [~, num_descriptors] = size(features_all);
    perm = randperm(num_descriptors);
    top = floor(num_descriptors*training_set_fraction);
    dictionary_learning_set = features_all(:, perm(1:top));
    
    % learn the dictionary using sparse coding
    dictionary = dict.sparse_coding(dictionary_learning_set, dictionary_size, iterations, lambda);
end

