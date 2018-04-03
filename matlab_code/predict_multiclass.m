function [labels, per_class_probabilities, per_class_predictions] = predict_multiclass(models, testing_vectors)
%PREDICT_MULTICLASS Summary of this function goes here
%   Detailed explanation goes here

    num_models = length(models);
    per_class_probabilities = zeros(size(testing_vectors, 1), num_models);
    per_class_predictions = zeros(size(testing_vectors, 1), num_models);
    
    for i = 1:num_models
        [predictions, probabilities] = svm.predict(models{i}, testing_vectors);

        per_class_probabilities(:, i) = probabilities(:, 2);
        per_class_predictions(:, i) = predictions;
    end

    % combine results
    [~, labels] = max(per_class_probabilities, [], 2);
end

