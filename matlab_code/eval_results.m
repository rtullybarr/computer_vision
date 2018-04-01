% Run this script to train the SVMs and see the classification results.
% Separated from main as a time-saving measure.

% some constants
num_species = 4;
% determines feature selection method to be used.

% Step 1: load images and set up class labels.
wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
guineaFowl = preprocess(get_image_filenames('guineaFowl', '*.jpg'), [256 256]);
hartebeest = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
giraffe = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);

species_names = ["wildebeest", "guineaFowl", "hartebeest", "giraffe"];
species_masks = [ones(length(wildebeest), 1); ones(length(guineaFowl), 1) .* 2; ones(length(hartebeest), 1) .* 3; ones(length(giraffe), 1) .* 4];

all_images = [wildebeest; guineaFowl; hartebeest; giraffe];

% Step 2: load intermediate results.
LBP_features = cell(num_species, 1);
load('intermediate_results/LBP_img_vec_wildebeest_dictsize_128_iter_10_lambda_26.mat');
LBP_features{1} = LBP_image_vectors;
load('intermediate_results/LBP_img_vec_guineaFowl_dictsize_128_iter_10_lambda_26.mat');
LBP_features{2} = LBP_image_vectors;
load('intermediate_results/LBP_img_vec_hartebeest_dictsize_128_iter_10_lambda_26.mat');
LBP_features{3} = LBP_image_vectors;
load('intermediate_results/LBP_img_vec_giraffe_dictsize_128_iter_10_lambda_26.mat');
LBP_features{4} = LBP_image_vectors;

SIFT_features = cell(num_species, 1);
load('intermediate_results/SIFT_img_vec_wildebeest_dictsize_128_iter_10_lambda_26');
SIFT_features{1} = SIFT_image_vectors;
load('intermediate_results/SIFT_img_vec_guineaFowl_dictsize_128_iter_10_lambda_26');
SIFT_features{2} = SIFT_image_vectors;
load('intermediate_results/SIFT_img_vec_hartebeest_dictsize_128_iter_10_lambda_26');
SIFT_features{3} = SIFT_image_vectors;
load('intermediate_results/SIFT_img_vec_giraffe_dictsize_128_iter_10_lambda_26');
SIFT_features{4} = SIFT_image_vectors;

% step 3: train one vs. all SVMs.
fprintf("Training SVMs.\n");
tic
% SVM training and testing
% permutation
perm = randperm(length(class_labels));

% 70% train, 30% test
split = floor(length(class_labels) * 0.7);

LBP_models = cell(num_species, 1);
SIFT_models = cell(num_species, 1);

LBP_scores = zeros(num_species, length(class_labels) - split);
SIFT_scores = zeros(num_species, length(class_labels) - split);
    
for i = 1:num_species
    % set up class labels
    class_labels = zeros(length(all_images), 1);
    class_labels(species_masks == positive_class) = 1;
    
    Y_train = class_labels(perm(1:split));
    Y_test = class_labels(perm(split + 1:end));

    % training set
    LBP_X_train = LBP_features{i}(perm(1:split), :);
    LBP_X_test = LBP_features{i}(perm(split + 1:end), :);

    SIFT_X_train = SIFT_features{i}(perm(1:split), :);
    SIFT_X_test = SIFT_features{i}(perm(split + 1:end), :);

    % LBP
    LBP_models{i} = svm.train(LBP_X_train, Y_train);
    % predictions - should be two scores for each image
    [predictions, scores] = svm.predict(LBP_models{i}, LBP_X_test);
    svm.score_predictions(predictions, Y_test)
    LBP_scores(i, :) = scores(:, 2);
    
    % SIFT
    SIFT_models{i} = svm.train(SIFT_X_train, Y_train);
    [predictions, scores] = svm.predict(LBP_models{i}, SIFT_X_test);
    svm.score_predictions(predictions, Y_test)
    SIFT_scores(i, :) = scores(:, 2);
end

% combine results
[~, LBP_classes] = max(LBP_scores, [], 1);
[~, SIFT_classes] = max(SIFT_scores, [], 1);

% get ground truth
ground_truth = species_masks(perm(split + 1:end), :);
confusionmat(LBP_classes, ground_truth);
confusionmat(SIFT_classes, ground_truth);