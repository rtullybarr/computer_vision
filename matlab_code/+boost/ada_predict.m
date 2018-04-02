% Produce classification output for a set of weighted weak classifiers 
% Input: 
% model -> cell array of (N feature types)x(T trials) SVMs
% alpha -> weights for T intemediate classifiers (weighted combo of T sets
%          of N SVMs)
% h_weights -> set of NxT weights for component SVMs
% testing set -> cell array of (M training samples) x (N feature types)
% mode -> "labels" (1/0) or "scores" (probability, 0->1)

function labels = ada_predict(model, alpha, h_weights, testing_set, mode)
    M = size(testing_set, 1);   % number of training samples
    N = size(testing_set, 2)-1; % number of different feature types, excluding class labels
    T = size(h_weights, 2);       % number of intermediate classifiers, or trials used in ada_train  
    H_test = zeros(M, T);       % set of predictions from each intermediate classifier  
    
    for t = 1:T
        H_test(:,t) = combo_predict(model{t}, h_weights(:,t), testing_set, N, mode);
    end
    
    labels(:,1) = weighted_vote(H_test, alpha, mode);
    if mode == "labels"
        labels(labels == -1) = 0;
    end
        
end

function predictions = combo_predict(models, weights, testing_set, N, mode)
    weak_predictions = zeros(size(testing_set,1),N);
    for f = 1:N
        Xtest = cell2mat(testing_set(:,f));
        weak_predictions(:,f) = predict(models{f}, Xtest);
    end
    predictions = weighted_vote(weak_predictions, weights, mode);
end

% Assign labels to data based on a weighted average of classifiers
function combo_labels = weighted_vote(trained_labels, weights, mode)
    if mode == "labels"
        combo_labels(:,1) = sign(sum(trained_labels*weights,2));
        combo_labels(combo_labels==0) = -1;
    else
        combo_labels(:,1) = sum(trained_labels*weights,2);
        combo_labels = combo_labels+abs(min(combo_labels));
        combo_labels = combo_labels/max(combo_labels);
    end
end
