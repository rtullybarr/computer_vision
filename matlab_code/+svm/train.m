function model = train(feature_descriptors, labels)
%TRAIN SVM
    model = fitcsvm(feature_descriptors, labels);
end

