% Prepare data for Gentleboosting
function [training_set, testing_set, Y_test] = ada_prep(LBP_image_vectors, SIFT_image_vectors, class_labels)
    fprintf("Fomatting image vectors for adaptive boosting\n");
    tic
    % permutation
    perm = randperm(length(class_labels));

    % 80% train, 20% test
    split = floor(length(class_labels) * 0.7);

    % training set
    LBP_X_train = LBP_image_vectors(perm(1:split), :);
    SIFT_X_train = SIFT_image_vectors(perm(1:split), :);
    Y_train = class_labels(perm(1:split));

        % split into pyramid levels
        LBP1_train = LBP_X_train(:,1:128);
        LBP2_train = LBP_X_train(:,129:641);
        LBP3_train = LBP_X_train(:,642:end);

        SIFT1_train = SIFT_X_train(:,1:128);
        SIFT2_train = SIFT_X_train(:,129:641);
        SIFT3_train = SIFT_X_train(:,642:end);

        % set class labels  = +1/-1 instead of 1/0
        Y_train(Y_train==0) = -1;


    % testing set
    LBP_X_test = LBP_image_vectors(perm(split + 1:end), :);
    SIFT_X_test = SIFT_image_vectors(perm(split + 1:end), :);
    Y_test = class_labels(perm(split + 1:end));

        % split into pyramid levels
        LBP1_test = LBP_X_test(:,1:128);
        LBP2_test = LBP_X_test(:,129:641);
        LBP3_test = LBP_X_test(:,642:end);

        SIFT1_test = SIFT_X_test(:,1:128);
        SIFT2_test = SIFT_X_test(:,129:641);
        SIFT3_test = SIFT_X_test(:,642:end);

    % format testing and training sets as cell arrays
    training_set = [num2cell(LBP1_train,2), num2cell(LBP2_train,2), num2cell(LBP3_train,2), ...
        num2cell(SIFT1_train,2), num2cell(SIFT2_train,2), num2cell(SIFT3_train,2),num2cell(Y_train,2)];

    testing_set = [num2cell(LBP1_test,2), num2cell(LBP2_test,2), num2cell(LBP3_test,2), ...
        num2cell(SIFT1_test,2), num2cell(SIFT2_test,2), num2cell(SIFT3_test,2)];
    toc
end

