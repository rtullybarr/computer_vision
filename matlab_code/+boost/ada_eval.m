function [precision, recall] = ada_eval(LBP_train, LBP_test, SIFT_train, SIFT_test, test_labels, train_labels, mode) 
tic
    % prep data for adaboost
    training_set = boost.ada_prep(LBP_train, SIFT_train, train_labels);
    testing_set = boost.ada_prep(LBP_test, SIFT_test, test_labels);
    
    % Create adaboosted classifer
    [ada_labels, model]= boost.ada_train(training_set, mode);
    if mode == "scores"
        ada_labels(ada_labels<mean(ada_labels)) = 0;
        ada_labels(ada_labels>=mean(ada_labels)) = 1;
    end
    
    % Evaluate classifier on test data
    TP = train_labels .* ada_labels; % both 1
    FP = ~train_labels .* ada_labels; % test_labels was 0 but predictions was 1
    FN = train_labels .* ~ada_labels; % test_labels was 1, but predictions was 0

    precision = sum(TP) / (sum(TP) + sum(FP));
    recall = sum(TP) / (sum(TP) + sum(FN));
    
    fprintf("ada_labels: precision = %f, recall = %f\n", precision, recall);
    
    % Evaluate classifier on training data with ada_predict
    [ada_labels2, scores] = boost.ada_predict(model, training_set);
    if mode == "scores"
        ada_labels2 = scores;
        ada_labels2(ada_labels2<mean(ada_labels2)) = 0;
        ada_labels2(ada_labels2>=mean(ada_labels2)) = 1;
    end
    
    TP = train_labels .* ada_labels2; % both 1
    FP = ~train_labels .* ada_labels2; % test_labels was 0 but predictions was 1
    FN = train_labels .* ~ada_labels2; % test_labels was 1, but predictions was 0

    precision = sum(TP) / (sum(TP) + sum(FP));
    recall = sum(TP) / (sum(TP) + sum(FN));
    
    fprintf("ada_labels2: precision = %f, recall = %f\n", precision, recall);
    
    % Test adaboosted classifier on test data
    [predictions, scores] = boost.ada_predict(model, testing_set);
    if mode == "scores"
        predictions = scores;
        predictions(predictions<mean(predictions)) = 0;
        predictions(predictions>=mean(predictions)) = 1;
    end

    % evaluate results
    TP = test_labels .* predictions; % both 1
    FP = ~test_labels .* predictions; % test_labels was 0 but predictions was 1
    FN = test_labels .* ~predictions; % test_labels was 1, but predictions was 0

    precision = sum(TP) / (sum(TP) + sum(FP));
    recall = sum(TP) / (sum(TP) + sum(FN));
    
    fprintf("predictions: precision = %f, recall = %f\n", precision, recall);
toc
end