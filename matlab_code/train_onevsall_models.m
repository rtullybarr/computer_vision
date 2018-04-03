function models = train_onevsall_models(image_vectors, train_indices, class_labels, num_species, mode)
%TRAIN_MULTICLASS_SVMS

    % SVM training and testing
    num_samples = size(image_vectors, 1);
    models = cell(num_species, 1);
    X_train = image_vectors(train_indices, :);
    
    if mode == "boost"
        len = size(X_train, 2)/2;
        LBP_train = X_train(1:len);
        SIFT_train = X_train(len+1:end);
    end
    
    for positive_class = 1:num_species
        % set up class labels
        binary_class_labels = zeros(num_samples, 1);
        binary_class_labels(class_labels == positive_class) = 1;

        Y_train = binary_class_labels(train_indices);

        if mode == "boost"
            ada_data = adaprep(LBP_train, SIFT_train, Y_train);
            [~, model] = boost.ada_train(ada_data, "labels");
        else  
            model = svm.train(X_train, Y_train);
            % Learns a function to convert from scores to probabilities.
            model = fitPosterior(model);
        end  
        
        models{positive_class} = model;
            
    end
end

