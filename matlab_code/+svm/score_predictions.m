function [precision, recall, confusion_matrix] = score_predictions(predictions, ground_truth)
%SCORE_PREDICTIONS compute precision, recall and confusion matrix for the
%given predictions.
    
    % evaluate results
    TP = ground_truth .* predictions; % both 1
    FP = ~ground_truth .* predictions; % ground_truth was 0 but predictions was 1
    FN = ground_truth .* ~predictions; % ground_truth was 1 but predictions was 0
    
    precision = sum(TP) / (sum(TP) + sum(FP));
    recall = sum(TP) / (sum(TP) + sum(FN));
    
    confusion_matrix = confusionmat(predictions, ground_truth);
end

