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

% permutation
num_samples = size(all_images, 1);
perm = randperm(num_samples);

% 70% train, 30% test
split = floor(num_samples * 0.7);

LBP_models = train_onevsall_models(LBP_image_vectors, perm(1:split), species_masks, 4);
SIFT_models = train_onevsall_models(SIFT_image_vectors, perm(1:split), species_masks, 4);
% boosted_models = ?;

% testing data
LBP_test = LBP_image_vectors(perm(split + 1:end), :);
SIFT_test = SIFT_image_vectors(perm(split + 1:end), :);
% get ground truth class labels for testing data
ground_truth = species_masks(perm(split + 1:end));

% make predictions
[LBP_labels, LBP_probabilities, LBP_predictions] = predict_multiclass(LBP_models, LBP_test);
[SIFT_labels, SIFT_probabilities, SIFT_predictions] = predict_multiclass(SIFT_models, SIFT_test);

% score predictions
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

% add fifth class for when all four models predict '0'
temp = sum(LBP_predictions, 2);
LBP_labels(~temp) = 5;

temp = sum(SIFT_predictions, 2);
SIFT_labels(~temp) = 5;

[LBP_precision_scores, LBP_recall_scores, LBP_confmat] = svm.score_predictions(ground_truth, LBP_labels);
[SIFT_precision_scores, SIFT_recall_scores, SIFT_confmat] = svm.score_predictions(ground_truth, SIFT_labels);

% find images that were classified as '5' (not recognized by any
% classifier) and display them.
err_indices = perm(split + 1 + find(LBP_labels == 5));
for i = 1:length(err_indices)
    figure; imshow(all_images{err_indices(i)});
end
