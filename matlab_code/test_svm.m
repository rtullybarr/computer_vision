% testing SVM on dummy data

X = rand(100, 16);
y = randi(2, 100, 1);
y(y == 2) = -1;

num_samples = size(X, 1);
split_point = round(num_samples*0.7);
perm = randperm(num_samples);
X_train = X(perm(1:split_point), :);
y_train = y(perm(1:split_point));
X_test = X(perm(split_point + 1:end), :);
y_test = y(perm(split_point + 1:end));

svm = svm.train(X_train, y_train);

prediction = svm.predict(svm, X_test);