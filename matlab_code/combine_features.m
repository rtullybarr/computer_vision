function boosted_features = combine_features(predictorData, classLabels)
%COMBINE_FEATURES combines the LBP and HOG features using the Gentle Boost algorithm

boosted_features = fitcensemble(predictorData,classLabels,'Method','GentleBoost', ...
    'NumLearningCycles',200);
end