function [precision, recall, confusion_matrix] = score_predictions(predictions, ground_truth)
%SCORE_PREDICTIONS compute precision, recall and confusion matrix for the
%given predictions.
    
    % evaluate results
    TP = ground_truth .* predictions; % both 1
    FP = ~ground_truth .* predictions; % Y_train was 0 but LBP_predictions was 1
    FN = ground_truth .* ~predictions; % Y_train was 1, but LBP_predictions was 0
    
    precision = sum(TP) / (sum(TP) + sum(FP));
    recall = sum(TP) / (sum(TP) + sum(FN));
    
    confusion_matrix = confusionmat(predictions, ground_truth);
    confusion_matrix = confusion_matrix ./ sum(confusion_matrix) .* 100;
end

