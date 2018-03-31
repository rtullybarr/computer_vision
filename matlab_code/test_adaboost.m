% Gentleboost 
LBP1_train = LBP_X_train(:,1:128);
LBP2_train = LBP_X_train(:,129:641);
LBP3_train = LBP_X_train(:,642:end);

SIFT1_train = SIFT_X_train(:,1:128);
SIFT2_train = SIFT_X_train(:,129:641);
SIFT3_train = SIFT_X_train(:,642:end);

LBP1_test = LBP_X_test(:,1:128);
LBP2_test = LBP_X_test(:,129:641);
LBP3_test = LBP_X_test(:,642:end);

SIFT1_test = SIFT_X_test(:,1:128);
SIFT2_test = SIFT_X_test(:,129:641);
SIFT3_test = SIFT_X_test(:,642:end);

Y_train(Y_train==0) = -1;

training_table = [num2cell(LBP1_train,2), num2cell(LBP2_train,2), num2cell(LBP3_train,2), ...
    num2cell(SIFT1_train,2), num2cell(SIFT2_train,2), num2cell(SIFT3_train,2),num2cell(Y_train,2)];

testing_table = [num2cell(LBP1_test,2), num2cell(LBP2_test,2), num2cell(LBP3_test,2), ...
    num2cell(SIFT1_test,2), num2cell(SIFT2_test,2), num2cell(SIFT3_test,2)];

tic
[ada_train, predictions] =  boost.myadaboost(training_table,  testing_table, 20);
tic
% evaluate results
predictions(predictions==-1) = 0;
TP = Y_test .* predictions; % both 1
FP = ~Y_test .* predictions; % Y_train was 0 but LBP_predictions was 1
FN = Y_test .* ~predictions; % Y_train was 1, but LBP_predictions was 0

precision = sum(TP) / (sum(TP) + sum(FP))
recall = sum(TP) / (sum(TP) + sum(FN))
