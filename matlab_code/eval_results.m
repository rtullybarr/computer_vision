% Run this script to train the SVMs and see the classification results.
% Separated from main as a time-saving measure.

% reproducibility
rng(1);

% some constants
num_species = 4;
% determines feature selection method to be used.

% Step 1: load images.
wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
guineaFowl = preprocess(get_image_filenames('guineaFowl', '*.jpg'), [256 256]);
hartebeest = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
giraffe = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);

species_names = ["wildebeest", "guineaFowl", "hartebeest", "giraffe"];
species_masks = [ones(length(wildebeest), 1); ones(length(guineaFowl), 1) .* 2; ones(length(hartebeest), 1) .* 3; ones(length(giraffe), 1) .* 4];

all_images = [wildebeest; guineaFowl; hartebeest; giraffe];

% Step 2: load intermediate results.
load('intermediate_results/LBP_img_vec_wildebeest_dictsize_128_iter_10_lambda_26.mat');
%load('intermediate_results/LBP_img_vec_guineaFowl_dictsize_128_iter_10_lambda_26.mat');
%load('intermediate_results/LBP_img_vec_hartebeest_dictsize_128_iter_10_lambda_26.mat');
%load('intermediate_results/LBP_img_vec_giraffe_dictsize_128_iter_10_lambda_26.mat');

load('intermediate_results/SIFT_img_vec_wildebeest_dictsize_128_iter_10_lambda_26.mat');
%load('intermediate_results/SIFT_img_vec_guineaFowl_dictsize_128_iter_10_lambda_26.mat');
%load('intermediate_results/SIFT_img_vec_hartebeest_dictsize_128_iter_10_lambda_26.mat');
%load('intermediate_results/SIFT_img_vec_giraffe_dictsize_128_iter_10_lambda_26.mat');

% step 3: train one vs. all SVMs.
fprintf("Training SVMs.\n");
tic
% SVM training and testing
% permutation
perm = randperm(length(all_images));

% 70% train, 30% test
split = floor(length(all_images) * 0.7);

% training data
LBP_X_train = LBP_image_vectors(perm(1:split), :);
LBP_X_test = LBP_image_vectors(perm(split + 1:end), :);

SIFT_X_train = SIFT_image_vectors(perm(1:split), :);
SIFT_X_test = SIFT_image_vectors(perm(split + 1:end), :);
    
LBP_models = cell(num_species, 1);
SIFT_models = cell(num_species, 1);

LBP_scores = zeros(length(all_images) - split, num_species);
SIFT_scores = zeros(length(all_images) - split, num_species);

LBP_predictions = zeros(length(all_images) - split, num_species);
SIFT_predictions = zeros(length(all_images) - split, num_species);

for positive_class = 1:num_species
    % set up class labels
    class_labels = zeros(length(all_images), 1);
    class_labels(species_masks == positive_class) = 1;
    
    Y_train = class_labels(perm(1:split));
    Y_test = class_labels(perm(split + 1:end));

    % LBP
    model = svm.train(LBP_X_train, Y_train);
    % Finds a function to convert from scores to probabilities.
    model = fitPosterior(model);
    [predictions, scores] = svm.predict(model, LBP_X_test);

    LBP_models{positive_class} = model;
    LBP_scores(:, positive_class) = scores(:, 2);
    LBP_predictions(:, positive_class) = predictions;
    
    % SIFT
    model = svm.train(SIFT_X_train, Y_train);
    model = fitPosterior(model);
    [predictions, scores] = svm.predict(model, SIFT_X_test);

    SIFT_models{positive_class} = model;
    SIFT_scores(:, positive_class) = scores(:, 2);
    SIFT_predictions(:, positive_class) = predictions;
end

% combine results
[~, LBP_labels] = max(LBP_scores, [], 2);
[~, SIFT_labels] = max(SIFT_scores, [], 2);

% add fifth class for if all four models predicted '0'?
% temp = sum(LBP_predictions, 2);
% LBP_labels(~temp) = 5;
% 
% temp = sum(SIFT_predictions, 2);
% SIFT_labels(~temp) = 5;

% get ground truth
ground_truth = species_masks(perm(split + 1:end));

% get results
[LBP_precision_scores, LBP_recall_scores, LBP_confmat] = svm.score_predictions(ground_truth, LBP_labels);
[SIFT_precision_scores, SIFT_recall_scores, SIFT_confmat] = svm.score_predictions(ground_truth, SIFT_labels);

% summarize results
LBP_precision = sum(LBP_precision_scores) / 4
LBP_recall = sum(LBP_recall_scores) / 4

SIFT_precision = sum(SIFT_precision_scores) / 4
SIFT_recall = sum(SIFT_recall_scores) / 4

% average accuracy
LBP_avg_accuracy = length(ground_truth(ground_truth == LBP_labels)) / length(ground_truth)
SIFT_avg_accuracy = length(ground_truth(ground_truth == SIFT_labels)) / length(ground_truth)
