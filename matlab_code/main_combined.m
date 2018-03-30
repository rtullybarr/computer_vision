%function [LBP_dictionary, LBP_model, LBP_X_test, SIFT_dictionary, SIFT_model, SIFT_X_test, Y_test] = main(dictionary_size, dictionary_iterations, lambda)
    rng('default');
    rng(1);
    dictionary_size = 128;
    dictionary_iterations = 10;
    lambda = 0.026;
    training_set_size = 20000;
    
%     fprintf("Reading and preprocessing images.\n");
%     tic
%     wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
%     guineaFowl = preprocess(get_image_filenames('guineaFowl', '*.jpg'), [256 256]);
%     hartebeest = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
%     giraffe = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);
%     
%     all_images = [wildebeest; guineaFowl; hartebeest; giraffe];
%     class_labels = [ones(length(wildebeest), 1); zeros(length(guineaFowl), 1); zeros(length(hartebeest), 1); zeros(length(giraffe), 1)];
%     toc

    fprintf("Extracting local binary patterns.\n");
    tic
    LBP_features = cell(length(all_images), 1);
    for i = 1:length(all_images)
        LBP_features{i} = feats.LBP(all_images{i});
    end
    toc
    
    fprintf("Calculating LS-LBP Descriptors\n");
    tic
    LSLBP_features = cell(length(all_images), 1);
    for i = 1:length(all_images)
        LSLBP_features{i} = feats.LS_descriptor(LBP_features{i});
    end
    toc

    fprintf("Extracting SIFT descriptors.\n");
    tic
    SIFT_features = cell(length(all_images), 1);
    for i = 1:length(all_images)
        SIFT_features{i} = double(feats.sift_features(all_images{i}));
    end
    toc
    
    fprintf("Calculating LS-SIFT Descriptors\n");
    tic
    LSSIFT_features = cell(length(all_images), 1);
    for i = 1:length(all_images)
        LSSIFT_features{i} = feats.LS_descriptor(SIFT_features{i});
    end
    toc
    
    combined_features = [LSLBP_features, LSSIFT_features]
    
    fprintf("Learning dictionary - combined.\n");
    tic
    combined_dictionary = dict.learn_dictionary(combined_features, training_set_size, dictionary_size, dictionary_iterations, lambda);
    toc
    
    fprintf("Assembling image vectors using SPM.\n");
    tic
    
    known quantity: the length of the image_vector returned by the SPM
    step is dict_size + 4*dict_size + 16*dict_size
    img_vector_len = 21*dictionary_size;

    combined_image_vectors = zeros(length(LBP_features), img_vector_len);
    for i = 1:length(LBP_features)
        % Use SPM to get a single image vector
        combined_image_vectors(i, :) = dict.spatial_pyramid_matching(combined_dictionary, combined_features{i}, lambda);
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
    combined_X_train = combined_image_vectors(perm(1:split), :);
    combined_X_test = combined_image_vectors(perm(split + 1:end), :);
    
    Y_train = class_labels(perm(1:split));
    Y_test = class_labels(perm(split + 1:end));
    
    
    % Training & Evaluation
     combined_model = svm.train(SIFT_X_train, Y_train);
     [combined_precision, combined_recall] = svm.evaluate_model(combined_model, combined_X_test, Y_test);

    % find misclassified images and display them.
    errors = find(Y_test ~= prediction);
    err_indices = perm(errors + split + 1);
    
    for i = 1:length(err_indices)
        imshow(all_images{err_indices(i)});
    end
    
    toc
%end
