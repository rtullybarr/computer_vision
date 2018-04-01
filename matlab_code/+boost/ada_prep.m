% Prepare data for Gentleboosting
function ada_data = ada_prep(LBP_samples, SIFT_samples, class_labels)
    fprintf("Fomatting image vectors for adaptive boosting\n");
    tic

    % split into pyramid levels
    LBP1 = LBP_samples(:,1:128);
    LBP2 = LBP_samples(:,129:641);
    LBP3 = LBP_samples(:,642:end);

    SIFT1 = SIFT_samples(:,1:128);
    SIFT2 = SIFT_samples(:,129:641);
    SIFT3 = SIFT_samples(:,642:end);

    % set class labels  = +1/-1 instead of 1/0
    class_labels(class_labels==0) = -1;

    % format data sets as cell arrays
    ada_data = [num2cell(LBP1,2), num2cell(LBP2,2), num2cell(LBP3,2), ...
        num2cell(SIFT1,2), num2cell(SIFT2,2), num2cell(SIFT3,2),num2cell(class_labels,2)];
    toc
end

