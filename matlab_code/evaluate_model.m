function [precision, recall] = evaluate_model(model, X_test, Y_test)
%EVALUATE_MODEL evaluates the provided model; returns precision and recall.

    predictions = test_svm(LBP_X_test);
    
    % evaluate results
    TP = Y_test .* predictions; % both 1
    FP = ~Y_test .* predictions; % Y_train was 0 but LBP_predictions was 1
    FN = Y_test .* ~predictions; % Y_train was 1, but LBP_predictions was 0
    
    precision = sum(TP) / (sum(TP) + sum(FP));
    recall = sum(TP) / (sum(TP) + sum(FN));
end

