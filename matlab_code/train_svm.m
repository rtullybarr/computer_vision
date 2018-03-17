function model = train_svm(feature_descriptors, labels)
%TRAIN SVM
    model = fitcsvm(feature_descriptors, labels);
end

