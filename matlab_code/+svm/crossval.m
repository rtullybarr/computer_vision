function results = crossval(model, folds)
%CROSSVAL cross validate the given model using the given training data.
    crossval_model = crossval(model, 'KFold', folds);
end

