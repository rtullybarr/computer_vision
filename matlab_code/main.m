%function [LBP_dictionary, LBP_model, LBP_X_test, SIFT_dictionary, SIFT_model, SIFT_X_test, Y_test] = main(dictionary_size, dictionary_iterations, lambda)

    dictionary_size = 128;
    dictionary_iterations = 10;
    lambda = 0.026;
    training_set_size = 20000;
    
    fprintf("Reading and preprocessing images.\n");
    tic
    wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
    guineaFowl = preprocess(get_image_filenames('guineaFowl', '*.jpg'), [256 256]);
    hartebeest = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
    giraffe = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);
    
    species_names = ["wildebeest", "guineaFowl", "hartebeest", "giraffe"];
    species_masks = [ones(length(wildebeest), 1); ones(length(guineaFowl), 1) .* 2; ones(length(hartebeest), 1) .* 3; ones(length(giraffe), 1) .* 4];
    
    all_images = [wildebeest; guineaFowl; hartebeest; giraffe];
    
    for positive_class = 3:4
        class_labels = zeros(length(all_images), 1);
        class_labels(species_masks == positive_class) = 1;

        fprintf("Extracting local binary patterns.\n");
        tic
        LBP_features = cell(length(all_images), 1);
        for i = 1:length(all_images)
            LBP_features{i} = feats.LBP(all_images{i});
        end
        toc

        fprintf("Learning dictionary - LBP.\n");
        tic
        LBP_dictionary = dict.learn_dictionary(LBP_features, training_set_size, dictionary_size, dictionary_iterations, lambda);
        toc

        fprintf("Extracting SIFT descriptors.\n");
        tic
        SIFT_features = cell(length(all_images), 1);
        for i = 1:length(all_images)
            SIFT_features{i} = double(feats.sift_features(all_images{i}));
        end
        toc

        fprintf("Learning dictionary - SIFT.\n");
        tic
        SIFT_dictionary = dict.learn_dictionary(SIFT_features, training_set_size, dictionary_size, dictionary_iterations, lambda);
        toc

        fprintf("Assembling image vectors using SPM.\n");
        tic

        % known quantity: the length of the image_vector returned by the SPM
        % step is dict_size + 4*dict_size + 16*dict_size
        img_vector_len = 21*dictionary_size;

        LBP_image_vectors = zeros(length(LBP_features), img_vector_len);
        for i = 1:length(LBP_features)
            % Use SPM to get a single image vector
            LBP_image_vectors(i, :) = dict.spatial_pyramid_matching(LBP_dictionary, LBP_features{i}, lambda);
        end

            % Use SPM to get a single image vector
            SIFT_image_vectors(i, :) = dict.spatial_pyramid_matching(SIFT_dictionary, SIFT_features{i}, lambda);
        end

        toc

        fprintf("Training SVMs.\n");
        tic
        % SVM training and testing
        % permutation
        perm = randperm(length(class_labels));

        % 80% train, 20% test
        split = floor(length(class_labels) * 0.7);

        % training set
        LBP_X_train = LBP_image_vectors(perm(1:split), :);
        LBP_X_test = LBP_image_vectors(perm(split + 1:end), :);

        SIFT_X_train = SIFT_image_vectors(perm(1:split), :);
        SIFT_X_test = SIFT_image_vectors(perm(split + 1:end), :);

        Y_train = class_labels(perm(1:split));
        Y_test = class_labels(perm(split + 1:end));

        % LBP
        LBP_model = svm.train(LBP_X_train, Y_train);
        [LBP_precision, LBP_recall] = svm.evaluate_model(LBP_model, LBP_X_test, Y_test);

        % SIFT
        SIFT_model = svm.train(SIFT_X_train, Y_train);
        [SIFT_precision, SIFT_recall] = svm.evaluate_model(SIFT_model, SIFT_X_test, Y_test);

        % find misclassified images and display them.
    %     errors = find(Y_test ~= prediction);
    %     err_indices = perm(errors + split + 1);
    %     
    %     for i = 1:length(err_indices)
    %         imshow(all_images{err_indices(i)});
    %     end
    
        % Save intermediate results: dictionary and image vectors.

        suffix = [char(species_names(positive_class)) '_dictsize_' num2str(dictionary_size) '_iter_' num2str(dictionary_iterations) '_lambda_' num2str(lambda)];
    
        save(['LPB_dict_' suffix], 'LBP_dictionary');
        save(['LPB_img_vec_' suffix], 'LBP_image_vectors');
        save(['SIFT_dict_' suffix], 'LBP_dictionary');
        save(['SIFT_img_vec_' suffix], 'LBP_image_vectors');
        toc
     end
%end
