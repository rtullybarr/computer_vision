%function [LBP_dictionary, LBP_model, LBP_X_test, SIFT_dictionary, SIFT_model, SIFT_X_test, Y_test] = main(dictionary_size, dictionary_iterations, lambda)

    dictionary_size = 128;
    dictionary_iterations = 10;
    lambda = 0.026;
    
    fprintf("Reading and preprocessing images.\n");
    tic
    wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
    guineaFowl = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
    
    all_images = [wildebeest; guineaFowl];
    class_labels = [ones(length(wildebeest), 1); zeros(length(guineaFowl), 1)];
    toc

    fprintf("Extracting local binary patterns.\n");
    tic
    LBP_features = cell(length(all_images), 1);
    for i = 1:length(all_images)
        LBP_features{i} = LBP(all_images{i});
    end
    toc
    
    training_set_size = 0.2;
    
    fprintf("Learning dictionary - LBP.\n");
    tic
    LBP_dictionary = learn_dictionary(LBP_features, training_set_size, dictionary_size, dictionary_iterations, lambda);
    toc

    fprintf("Extracting SIFT descriptors.\n");
    tic
    SIFT_features = cell(length(all_images), 1);
    for i = 1:length(all_images)
        SIFT_features{i} = double(sift_features(all_images{i}));
    end
    toc
    
    fprintf("Learning dictionary - SIFT.\n");
    tic
    SIFT_dictionary = learn_dictionary(SIFT_features, training_set_size, dictionary_size, dictionary_iterations, lambda);
    toc
    
    fprintf("Assembling image vectors using SPM.\n");
    tic
    
    % known quantity: the length of the image_vector returned by the SPM
    % step is dict_size + 4*dict_size + 16*dict_size
    img_vector_len = 21*dictionary_size;
    
    LBP_image_vectors = zeros(length(LBP_features), img_vector_len);
    for i = 1:length(LBP_features)
        % Use SPM to get a single image vector
        LBP_image_vectors(i, :) = spatial_pyramid_matching(LBP_dictionary, LBP_features{i}, lambda);
    end
    
    SIFT_image_vectors = zeros(length(LBP_features), img_vector_len);
    for i = 1:length(LBP_features)
        % Use SPM to get a single image vector
        SIFT_image_vectors(i, :) = spatial_pyramid_matching(SIFT_dictionary, SIFT_features{i}, lambda);
    end
    
    toc
    
    fprintf("Training SVMs.\n");
    tic
    % SVM training and testing
    % permutation
    perm = randperm(length(class_labels));
    
    % 80% train, 20% test
    split = floor(length(class_labels) * 0.8);
    
    % training set
    LBP_X_train = LBP_image_vectors(perm(1:split), :);
    LBP_X_test = LBP_image_vectors(perm(split + 1:end), :);
    
    SIFT_X_train = SIFT_image_vectors(perm(1:split), :);
    SIFT_X_test = SIFT_image_vectors(perm(split + 1:end), :);
    
    Y_train = class_labels(perm(1:split), :);
    Y_test = class_labels(perm(split + 1:end), :);
    
    % LBP
    LBP_model = train_svm(LBP_X_train, Y_train);
    [LBP_precision, LBP_recall] = evaluate_model(LBP_model, LBP_X_test, Y_test);
    
    % SIFT
    SIFT_model = train_svm(SIFT_X_train, Y_train);
    [SIFT_precision, SIFT_recall] = evaluate_model(SIFT_model, SIFT_X_test, Y_test);
    
    toc
%end
