function [precision, recall, confusion_matrix] = score_predictions(ground_truth, predictions)
%SCORE_PREDICTIONS compute precision, recall and confusion matrix for the
%given predictions.

    % compute the confusion matrix
    confusion_matrix = confusionmat(ground_truth, predictions);
    num_species = length(confusion_matrix);
    
    % compute per-class precision and recall
    % row: ground truth
    precision = zeros(num_species, 1);
    recall = zeros(num_species, 1);
    
    for i = 1:num_species
        % TP for class i = entry (i, i)
        TP = confusion_matrix(i, i);
        % FN for class i = row i without entry (i, i)
        FN = sum(confusion_matrix(i, :)) - TP;
        % FP for class i = column i without entry (i, i)
        FP = sum(confusion_matrix(:, i)) - TP;
        
        precision(i) = TP / (TP + FP);
        recall(i) = TP / (TP + FN);
    end
end

