% CROSS_VALIDATE_ADA
% Performs cross validation for adaboost 
% 
% INPUTS
% k - k fold validation (# of data partitions to use)
% class_labels - class label matrix with the positive class set to 1 and
%       negatives to 0
% LBP_image_vectors - training set of LBP image vectors
% SIFT_image_vectors - training set of SIFT image vectors
% 
% OUTPUT
% trained_model - cell array with {ada_labels, h_model, h_weights,alpha}

function trained_model = cross_validate_ada(k, class_labels,...
    LBP_image_vectors, SIFT_image_vectors)

    partition = 1/k;
    ada_labels =  cell(1,k);
    h_model = cell(1,k);
    h_weights = cell(1,k);
    alpha = cell(1,k);
    
    % Get random order for cross validation
    perm = randperm(length(class_labels));

    % Split interval (dependent on k)
    split_interval = floor(length(class_labels) * partition);
    
    for i = 1:k
        % Reorder image vectors and take a subset
        LBP_samples = LBP_image_vectors(perm(1:end), :);
        LBP_samples((i-1)*split_interval+1:i*split_interval, :) = [];
        
        SIFT_samples = SIFT_image_vectors(perm(1:end), :);
        SIFT_samples((i-1)*split_interval+1:i*split_interval, :) = [];
        
        labels = class_labels(perm(1:end), :);
        labels((i-1)*split_interval+1:i*split_interval, :) = [];
        
        % Format data for adaboost training
        ada_data = boost.ada_prep(LBP_samples, SIFT_samples, labels);


    % Train data
    [ada_labels{1,i}, h_model{1,i}, h_weights{1,i}, alpha{1,i}]= boost.ada_train(ada_data);
        
    end
    
    trained_model = {ada_labels, h_model, h_weights,alpha};


end
