function [precision, recall, confusion_matrix] = score_predictions(predicted, ground_truth)
%SCORE_PREDICTIONS compute precision, recall and confusion matrix for the
%given predictions.
    
    % evaluate results
    TP = Y_test .* predictions; % both 1
    FP = ~Y_test .* predictions; % Y_train was 0 but LBP_predictions was 1
    FN = Y_test .* ~predictions; % Y_train was 1, but LBP_predictions was 0
    
    precision = sum(TP) / (sum(TP) + sum(FP));
    recall = sum(TP) / (sum(TP) + sum(FN));
    
    confusion_matrix = confusionmat(predicted, ground_truth);
    confusion_matrix = confusion_matrix ./ sum(confusion_matrix) .* 100;
end

