function [labels, per_class_probabilities, per_class_predictions] = predict_multiclass(models, testing_vectors, mode)
%PREDICT_MULTICLASS Summary of this function goes here
%   Detailed explanation goes here

    num_models = length(models);
    per_class_probabilities = zeros(size(testing_vectors, 1), num_models);
    per_class_predictions = zeros(size(testing_vectors, 1), num_models);
    
    if mode == "boost"
        len = size(testing_vectors, 2)/2;
        LBP_test = testing_vectors(1:len);
        SIFT_test = testing_vectors(len+1:end);
        testing_set = boost.ada_prep(LBP_test, SIFT_test, zeros(size(testing_vectors,1),1));
    end
    
    for i = 1:num_models
        if mode == "boost"
            [predictions, probabilities] = boost.ada_predict(models{i}, testing_set);
        else
            [predictions, probabilities] = svm.predict(models{i}, testing_vectors);
        end

        per_class_probabilities(:, i) = probabilities(:, 2);
        per_class_predictions(:, i) = predictions;
    end

    % combine results
    [~, labels] = max(per_class_probabilities, [], 2);
end

