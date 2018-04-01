
function [precision, recall, LBP_test_set] = cross_validate_svm(species_masks, class_labels,...
    positive_class, LBP_image_vectors, SIFT_image_vectors, k)

    partition = 1/k;
    precision = zeros(k:1);
    recall = zeros(k:1);
    
    % SVM training and testing
    % permutation
    perm = randperm(length(class_labels));

    % Split interval (dependent on k)
    split_interval = floor(length(class_labels) * partition);
    
    %loop
    
    % Prepare data for adaboost
    LBP_X_train = LBP_image_vectors(perm(1:split), :);
    SIFT_X_train = SIFT_image_vectors(perm(1:split), :);
    
    

    SIFT1_train = SIFT_X_train(:,1:128);
    SIFT2_train = SIFT_X_train(:,129:641);
    SIFT3_train = SIFT_X_train(:,642:end);
    
    
    Y_train(Y_train==0) = -1;

    training_table = [num2cell(LBP1_train,2), num2cell(LBP2_train,2), num2cell(LBP3_train,2), ...
        num2cell(SIFT1_train,2), num2cell(SIFT2_train,2), num2cell(SIFT3_train,2),num2cell(Y_train,2)];
    
    for i = 1:k
        % Reorder image vectors and take a subset
        LBP_X_train = LBP_image_vectors(perm(1:end), :);
        LBP_X_train((i-1)*split+1:i*split) = [];
        
        % Prepare data for ada boost
        LBP1_train = LBP_X_train(:,1:128);
        LBP2_train = LBP_X_train(:,129:641);
        LBP3_train = LBP_X_train(:,642:end);

        SIFT1_train = SIFT_X_train(:,1:128);
        SIFT2_train = SIFT_X_train(:,129:641);
        SIFT3_train = SIFT_X_train(:,642:end);
        
        Y_train = class_labels(perm(1:split));
        
        [precision(i), recall(i)] = adaboost_train();
    end
    
    % training set
    LBP_X_train = LBP_image_vectors(perm(1:split_interval), :);
    LBP_X_test = LBP_image_vectors(perm(split_interval + 1:end), :);

    Y_train = class_labels(perm(1:split_interval));
    Y_test = class_labels(perm(split_interval + 1:end));



end
% ___________________________________

% wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
% guineaFowl = preprocess(get_image_filenames('guineaFowl', '*.jpg'), [256 256]);
% hartebeest = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
% giraffe = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);
% 
% species_names = ["wildebeest", "guineaFowl", "hartebeest", "giraffe"];
% species_masks = [ones(length(wildebeest), 1); ones(length(guineaFowl), 1) .* 2; ones(length(hartebeest), 1) .* 3; ones(length(giraffe), 1) .* 4];
% 
% all_images = [wildebeest; guineaFowl; hartebeest; giraffe];
positive_class = 2;

class_labels = zeros(length(all_images), 1);
class_labels(species_masks == positive_class) = 1;

[m, n] = size(LBP_image_vectors);

% SVM training and testing
% permutation
perm = randperm(length(class_labels));

% 80% train, 20% test
split = floor(length(class_labels) * 0.7);

% training set
LBP_X_train = LBP_image_vectors(perm(1:split), :);
LBP_X_test = LBP_image_vectors(perm(split + 1:end), :);

Y_train = class_labels(perm(1:split));
Y_test = class_labels(perm(split + 1:end));
        
% LBP
fprintf("Training LBP SVM.\n");
tic
LBP_model = svm.train(LBP_X_train, Y_train, 'crossval');
toc
fprintf("Evaluate model\n");

LBP_precision = zeros(1, 10);
LBP_recall = zeros(1, 10);

tic
for i = 1:10
    [LBP_precision(i), LBP_recall(i)] = svm.evaluate_model(LBP_model.Trained{i}, LBP_X_test, Y_test);
end
toc
