function [precision, recall] = ada_eval(LBP_image_vectors, SIFT_image_vectors, class_labels)
tic
    % prep data for adaboost
    [training_set, testing_set, Y_test] = ada_prep(LBP_image_vectors, SIFT_image_vectors, class_labels);
    
    % Create adaboosted classifer
    [ada_labels, h_model, h_weights, alpha]= boost.ada_train(training_set);
    
    % Test adaboosted classifier
    predictions = ada_predict(h_model, alpha, h_weights, testing_set);

    % evaluate results
    TP = Y_test .* predictions; % both 1
    FP = ~Y_test .* predictions; % Y_train was 0 but LBP_predictions was 1
    FN = Y_test .* ~predictions; % Y_train was 1, but LBP_predictions was 0

    precision = sum(TP) / (sum(TP) + sum(FP))
    recall = sum(TP) / (sum(TP) + sum(FN))
toc
end