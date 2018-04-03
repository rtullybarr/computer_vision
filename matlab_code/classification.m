% Run this script to train the SVMs and see the classification results.
% Separated from main as a time-saving measure.

% reproducibility
rng(1);

% some constants
num_species = 4;

% Step 1: load images.
wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
guineaFowl = preprocess(get_image_filenames('guineaFowl', '*.jpg'), [256 256]);
hartebeest = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
giraffe = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);

species_names = ["wildebeest", "guineaFowl", "hartebeest", "giraffe"];
species_masks = [ones(length(wildebeest), 1); ones(length(guineaFowl), 1) .* 2; ones(length(hartebeest), 1) .* 3; ones(length(giraffe), 1) .* 4];

all_images = [wildebeest; guineaFowl; hartebeest; giraffe];

% Step 2: load intermediate results.
load('intermediate_results/trial_1_dictsize_128_iter_10_lambda_26.mat');

% step 3: train one vs. all SVMs.
fprintf("Training SVMs.\n");
tic

% permutation
num_samples = size(all_images, 1);
perm = randperm(num_samples);

% 70% train, 30% test
split = floor(num_samples * 0.7);

validation_indices = perm(1:split);
final_test_indices = perm(split + 1:end);

% 5-fold cross validation
partition = 1/5;
% Split interval (dependent on k)
split_interval = floor(length(validation_indices) * partition);

varNames = {'precision', 'recall', 'accuracy'};
rowNames = {'cv1', 'cv2', 'cv3', 'cv4', 'cv5', 'final'};
LBP_results = table(zeros(6,1), zeros(6,1), zeros(6,1),'VariableNames',varNames,'RowNames',rowNames);
SIFT_results = table(zeros(6,1), zeros(6,1), zeros(6,1),'VariableNames',varNames,'RowNames',rowNames);
BOOST_results = table(zeros(6,1), zeros(6,1), zeros(6,1),'VariableNames',varNames,'RowNames',rowNames);

for k = 1:6
    if k == 6
        train_indices = validation_indices;
        test_indices = final_test_indices;
    else
        train_indices = validation_indices;
        train_indices((k-1)*split_interval+1:k*split_interval) = [];
        test_indices = validation_indices((k-1)*split_interval+1:k*split_interval);
    end
    
    LBP_models = train_onevsall_models(LBP_image_vectors, train_indices, species_masks, 4, "single");
    SIFT_models = train_onevsall_models(SIFT_image_vectors, train_indices, species_masks, 4, "single");
    % Train AdaBoosted model for combined features
    all_image_vectors = [LBP_image_vectors, SIFT_image_vectors];
    BOOST_models = train_onevsall_models(all_image_vectors, train_indices, species_masks, 4, "boost");

    % testing data
    LBP_test = LBP_image_vectors(test_indices, :);
    SIFT_test = SIFT_image_vectors(test_indices, :);
    BOOST_test = all_image_vectors(test_indices, :);
    % get ground truth class labels for testing data
    ground_truth = species_masks(test_indices);

    % make predictions
    [LBP_labels, LBP_probabilities, LBP_predictions] = predict_multiclass(LBP_models, LBP_test, "single");
    [SIFT_labels, SIFT_probabilities, SIFT_predictions] = predict_multiclass(SIFT_models, SIFT_test, "single");
    [BOOST_labels, BOOST_probabilities, BOOST_predictions] = predict_multiclass(BOOST_models, BOOST_test, "boost");

    % add fifth class for when all four models predict '0'
    temp = sum(LBP_predictions, 2);
    LBP_labels(~temp) = 5;

    temp = sum(SIFT_predictions, 2);
    SIFT_labels(~temp) = 5;
    
    temp = sum(BOOST_predictions, 2);
    BOOST_labels(~temp) = 5;
    
    % score predictions
    [LBP_precision_scores, LBP_recall_scores, LBP_confmat] = svm.score_predictions(ground_truth, LBP_labels);
    [SIFT_precision_scores, SIFT_recall_scores, SIFT_confmat] = svm.score_predictions(ground_truth, SIFT_labels);
    [BOOST_precision_scores, BOOST_recall_scores, BOOST_confmat] = svm.score_predictions(ground_truth, BOOST_labels);
    
    % summarize results (excluding precision/recall for class 5, which are
    % 0/NaN by definition)
    LBP_results.precision(k) = sum(LBP_precision_scores(1:4)) / 4;
    LBP_results.recall(k) = sum(LBP_recall_scores(1:4)) / 4;

    SIFT_results.precision(k) = sum(SIFT_precision_scores(1:4)) / 4;
    SIFT_results.recall(k) = sum(SIFT_recall_scores(1:4)) / 4;
    
    BOOST_results.precision(k) = sum(BOOST_precision_scores(1:4)) / 4;
    BOOST_results.recall(k) = sum(BOOST_recall_scores(1:4)) / 4;

    % accuracy
    LBP_results.accuracy(k) = length(ground_truth(ground_truth == LBP_labels)) / length(ground_truth);
    SIFT_results.accuracy(k) = length(ground_truth(ground_truth == SIFT_labels)) / length(ground_truth);
    BOOST_results.accuracy(k) = length(ground_truth(ground_truth == BOOST_labels)) / length(ground_truth);
    
    if k == 6
        disp('---- Final Testing Results ----')
        
        disp('---- Local Binary Patterns Alone ----')
        disp(LBP_results);
        % mean and standard deviation
        disp('cross-validated precision')
        disp(['mean=' num2str(mean(LBP_results.precision(1:5))) '  std=' num2str(std(LBP_results.precision(1:5)))])
        disp('cross-validated recall')
        disp(['mean=' num2str(mean(LBP_results.recall(1:5))) '  std=' num2str(std(LBP_results.recall(1:5)))])
        disp('cross-validated accuracy')
        disp(['mean=' num2str(mean(LBP_results.accuracy(1:5))) '  std=' num2str(std(LBP_results.accuracy(1:5)))])
        
        disp('---- SIFT Alone ----')
        disp(SIFT_results);
        disp('cross-validated precision')
        disp(['mean=' num2str(mean(SIFT_results.precision(1:5))) '  std=' num2str(std(SIFT_results.precision(1:5)))])
        disp('cross-validated recall')
        disp(['mean=' num2str(mean(SIFT_results.recall(1:5))) '  std=' num2str(std(SIFT_results.recall(1:5)))])
        disp('cross-validated accuracy')
        disp(['mean=' num2str(mean(SIFT_results.accuracy(1:5))) '  std=' num2str(std(SIFT_results.accuracy(1:5)))])
        
        disp('---- Combined, Boosted results ----')
        disp(BOOST_results);
        disp('cross-validated precision')
        disp(['mean=' num2str(mean(BOOST_results.precision(1:5))) '  std=' num2str(std(BOOST_results.precision(1:5)))])
        disp('cross-validated recall')
        disp(['mean=' num2str(mean(BOOST_results.recall(1:5))) '  std=' num2str(std(BOOST_results.recall(1:5)))])
        disp('cross-validated accuracy')
        disp(['mean=' num2str(mean(BOOST_results.accuracy(1:5))) '  std=' num2str(std(BOOST_results.accuracy(1:5)))])
        
        
    end

    % find images that were not recognized by any classifier and save them.
    if k == 6
        LBP_err_indices = perm(split + find(LBP_labels == 5));
        for i = 1:length(LBP_err_indices)
            imwrite(all_images{LBP_err_indices(i)}, ['misclassified_images/LBP_miss_' num2str(LBP_err_indices(i)) '.jpg']);
        end
        
        SIFT_err_indices = perm(split + find(SIFT_labels == 5));
        for i = 1:length(SIFT_err_indices)
            imwrite(all_images{SIFT_err_indices(i)}, ['misclassified_images/SIFT_miss_' num2str(SIFT_err_indices(i)) '.jpg']);
        end
        
        BOOST_err_indices = perm(split + find(SIFT_labels == 5));
        for i = 1:length(BOOST_err_indices)
            imwrite(all_images{BOOST_err_indices(i)}, ['misclassified_images/BOOST_miss_' num2str(BOOST_err_indices(i)) '.jpg']);
        end
    end
end

% Display Confusion Matrices 
plot_CM(LBP_confmat, 4);
title('LBP Confusion Matrix')
plot_CM(SIFT_confmat, 4);
title('SIFT Confusion Matrix')
plot_CM(BOOST_confmat, 4);
title('AdaBoost Confusion Matrix')
