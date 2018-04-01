function [precision, recall] = ada_eval(LBP_train, LBP_test, SIFT_train, SIFT_test, test_labels, train_labels) 
tic
    % prep data for adaboost
    training_set = ada_prep(LBP_train, SIFT_train, train_labels);
    testing_set = ada_prep(LBP_test, SIFT_test, test_labels);
    
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