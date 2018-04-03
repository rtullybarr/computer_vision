% Run this script to train the SVMs and see the classification results.
% Separated from main as a time-saving measure.

% reproducibility
rng(1);

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
% temp = cell(num_species, 1);
%load('intermediate_results/LBP_img_vec_wildebeest_dictsize_128_iter_10_lambda_26.mat');
% temp{1} = LBP_image_vectors;
 load('intermediate_results/LBP_img_vec_guineaFowl_dictsize_128_iter_10_lambda_26.mat');
% temp{2} = LBP_image_vectors;
% load('intermediate_results/LBP_img_vec_hartebeest_dictsize_128_iter_10_lambda_26.mat');
% temp{3} = LBP_image_vectors;
% load('intermediate_results/LBP_img_vec_giraffe_dictsize_128_iter_10_lambda_26.mat');
% temp{4} = LBP_image_vectors;
% LBP_image_vectors = vertcat(temp{:});

% temp = cell(num_species, 1);
%load('intermediate_results/SIFT_img_vec_wildebeest_dictsize_128_iter_10_lambda_26.mat');
% temp{1} = SIFT_image_vectors;
 load('intermediate_results/SIFT_img_vec_guineaFowl_dictsize_128_iter_10_lambda_26.mat');
% temp{2} = SIFT_image_vectors;
% load('intermediate_results/SIFT_img_vec_hartebeest_dictsize_128_iter_10_lambda_26.mat');
% temp{3} = SIFT_image_vectors;
% load('intermediate_results/SIFT_img_vec_giraffe_dictsize_128_iter_10_lambda_26.mat');
% temp{4} = SIFT_image_vectors;
% SIFT_image_vectors = vertcat(temp{:});

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
    [LBP_precision, LBP_recall, LBP_confusion_matrix] = svm.score_predictions(predictions, Y_test)
    
    LBP_models{positive_class} = model;
    LBP_scores(:, positive_class) = scores(:, 2);
    LBP_predictions(:, positive_class) = predictions;
    
    % SIFT
    model = svm.train(SIFT_X_train, Y_train);
    model = fitPosterior(model);
    [predictions, scores] = svm.predict(model, SIFT_X_test);
    [SIFT_precision, SIFT_recall, SIFT_confusion_matrix] = svm.score_predictions(predictions, Y_test)
    
    SIFT_models{positive_class} = model;
    SIFT_scores(:, positive_class) = scores(:, 2);
    SIFT_predictions(:, positive_class) = predictions;
end

% combine results
[~, LBP_classes] = max(LBP_scores, [], 2);
[~, SIFT_classes] = max(SIFT_scores, [], 2);

% get ground truth
ground_truth = species_masks(perm(split + 1:end));

% display confusion matrices
LBP_matrix = confusionmat(ground_truth,LBP_classes);
%plot_confusion_matrix(LBP_matrix);
plot_CM(LBP_matrix);
title('LBP Confusion Matrix')

SIFT_matrix = confusionmat( ground_truth, SIFT_classes);
plot_CM(SIFT_matrix);
title('SIFT Confusion Matrix')