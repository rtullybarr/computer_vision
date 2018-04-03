function [models, crossval_results] = train_onevsall_models(image_vectors, train_indices, class_labels, num_species)
%TRAIN_MULTICLASS_SVMS

    % SVM training and testing
    num_samples = size(image_vectors, 1);
    models = cell(num_species, 1);
    crossval_results = zeros(num_species, 1);
    X_train = image_vectors(train_indices, :);
    
    for positive_class = 1:num_species
        % set up class labels
        binary_class_labels = zeros(num_samples, 1);
        binary_class_labels(class_labels == positive_class) = 1;

        Y_train = binary_class_labels(train_indices);

        % LBP
        crossval_model = svm.train(X_train, Y_train, 'crossval');
        crossval_results(positive_class) = kfoldLoss(crossval_model);
        % Learns a function to convert from scores to probabilities.
        model = svm.train(X_train, Y_train, 'simple');
        model = fitPosterior(model);
        models{positive_class} = model;
    end
end

