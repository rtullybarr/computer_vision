function [ada_train, ada_test]= myadaboost(training_table, testing_table, trials)
% AdaBoost function 
% training_table-> input: training set
%           feature_1 feature_2 ... feature_n class_labels
%  sample 1   1xN1      1xN2    ...     1xNn       +1/-1 
%  sample 2   1xN1      1xN2    ...     1xNn       +1/-1
%   ...        ...      ...     ...     ...         ...
%  sample M   1xN1      1xN2    ...     1xNn       +1/-1 
%   
% testing_table-> input: testing set (as above, w/o class_labels column)
% trials-> input: number of trials to test (# intermediate classifiers)
% ada_train-> label: training set
% ada_test-> label: testing set

% Choosen Weak classifier: SVM
% training_table = [num2cell(f1,2), num2cell(f2,2), num2cell(f3,2), num2cell(class_labels,2)]

% Initialize Variables
M = size(training_table,1);     % number of training samples
N = size(training_table,2)-1;   % number of feature types
T = trials;                     % number of training trials
D =(1/M)*ones(M,1);             % initial training sample weights  

svm_model = cell(1, N);  % weak learners   
svm_train = zeros(M,N);  % weak learner training outputs  
%eps_w = zeros(1,N);      % classification errors of weak learners

h_model = cell(T, 1);    % intermediate classifiers
h_train = zeros(M,T);    % intermediate classifier training outputs 
h_test = zeros(size(testing_table,1),T);     % intermediate classifier test outputs  
eps_h = zeros(T,1);      % intermediate classifier errors   
alpha = zeros(T,1);      % intermediate classifier weights

%difficult_data = [];     % data with weights above a threshold, difficult to classify, and selected more often over the boosting process
S = cell(M,N+1);          % weighted training set
sigma = 24;
for t = 1:T
    % train an intermediate classifier h(t)
    tic
    fprintf("Trial %d.\n", t);
    % 1. generate new weighted training set St using Dt
            p_min=min(D);
            p_max=max(D);

            for i=1:M
                % determine which samples have a higher weight (i.e. difficult to
                % classify), and add those to matrix difficult_data

                p = (p_max-p_min)*rand(1) + p_min;
                if D(i)>=p
                    difficult_data(i,:)=training_table(i,:);
                end

                % select random row of difficult_data, add to training set S 
                n=randi(size(difficult_data,1)); 
                S(i,:) = difficult_data(n,:);
            end

            % Separate features and class labels for training
            Sx=S(:,1:end-1);
            Y=cell2mat(S(:,end));

    % 2. train an svm classifier for each feature type f
        for f = 1:N
            fprintf("Training SVM %d out of %d\n", f, N);
            X = cell2mat(Sx(:,f));
            svm_model{f}=fitcsvm(X,Y,'KernelFunction','rbf', 'KernelScale', sigma);
            svm_train(:,f)=predict(svm_model{f}, X);
        end 

    % 3. intermediate classifier h(t) = linear combination of svm classifiers
        h_model{t} = svm_model;
        h_train(:,t) = round(sum(svm_train*(1/N),2));
    
    % 3.5 apply intermediate classifier  h(t) to test data 
        h_test(:,t) = combo_predict(h_model{t}, testing_table,N);

    % 4. calculate classification error for intermediate classifier h(t)
        for i=1:M
            if (h_train(i,t)~=Y(i))
                eps_h(t)=eps_h(t)+D(i); 
            end  
        end
        fprintf("Epsilon %d = %f.\n", t, eps_h(t));

    % 5. Calculate intermediate classifier weight
         alpha(t)=0.5*log((1-eps_h(t))/eps_h(t));
         fprintf("Alpha %d = %f.\n", t, alpha(t));

    % 6. Update training sample weights
         D=D.*exp((-1).*Y.*alpha(t).*h_train(t));
         D=D./sum(D);
    sigma = sigma-2;
    toc
end
    
% final vote
ada_train(:,1)=h_train*alpha;
ada_train(ada_train<=5) = 0;
ada_train(ada_train>5) = 1;

% for test set
ada_test(:,1)=h_test*alpha;
ada_test(ada_test<=5) = 0;
ada_test(ada_test>5) = 1;
end

function predictions = combo_predict(models, testing_table, N)
    weak_predictions = zeros(size(testing_table,1),N);
    for f = 1:N
        Xtest = cell2mat(testing_table(:,f));
        weak_predictions(:,f) = predict(models{f}, Xtest);
    end
    predictions = round(sum(weak_predictions*(1/N),2));
end