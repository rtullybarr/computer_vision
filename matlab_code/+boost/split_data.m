function [LBP_train, LBP_test, SIFT_train, SIFT_test, test_labels, train_labels] = split_data(LBP_image_vectors, SIFT_image_vectors, class_labels)% permutation
        
        perm = randperm(length(class_labels));

        % 70% train, 30% test
        split = floor(length(class_labels) * 0.7);

        % training and testing sets
        LBP_train = LBP_image_vectors(perm(1:split), :);
        LBP_test = LBP_image_vectors(perm(split + 1:end), :);

        SIFT_train = SIFT_image_vectors(perm(1:split), :);
        SIFT_test = SIFT_image_vectors(perm(split + 1:end), :);

        train_labels = class_labels(perm(1:split));
        test_labels = class_labels(perm(split + 1:end));
        
end