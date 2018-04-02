function model = train(feature_descriptors, labels, training_type)
%TRAIN SVM
%   training_type = 'single' or 'crossval'

    model = fitcsvm(feature_descriptors, labels);
    
    if nargin > 2 && strcmp(training_type, 'crossval')
    
        % Cross validate SVM - default 10 fold cross validation
        model = crossval(model);
    end
    

end

