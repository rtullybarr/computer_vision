function [ada_labels, h_model, h_weights, alpha]= ada_train(training_set)
% AdaBoost function 
% training_table-> input: training set, cell array formatted as below
%           feature_1 feature_2 ... feature_n class_labels
%  sample 1   1xN1      1xN2    ...     1xNn       +1/-1 
%  sample 2   1xN1      1xN2    ...     1xNn       +1/-1
%   ...        ...      ...     ...     ...         ...
%  sample M   1xN1      1xN2    ...     1xNn       +1/-1 
%   
% ada_labels-> labels for training set, 1/0 for yes/no
% h_model -> intemediate classifiers, set of (N feature types)x(T trials) SVMs
% h_weights -> NxT weights for SVMs in h_model
% alpha -> T weights for intemediate classifers h_model(:,1:T)
% Choosen Weak classifier: SVM

% Initialize Variables
M = size(training_set,1);     % number of training samples
N = size(training_set,2)-1;   % number of feature types
T = 20    ;                   % maximum number of training trials
D =(1/M)*ones(M,1);           % initial training sample weights  

svm_model = cell(1, N);       % weak learners   
svm_labels = zeros(M,N);       % weak learner training outputs  

h_model = cell(T, 1);         % intermediate classifiers
h_labels = zeros(M,T);         % intermediate classifier training outputs 
h_weights = zeros(N,T);       % intermediate classifier weights   
alpha = zeros(T,1);           % final classifier weights

difficult_data = cell(0);     % data with weights above a threshold, difficult to classify, and selected more often over the boosting process
S = cell(M,N+1);              % weighted training set

% Calculate the average minimal distance between any two training samples
sigma_min_all = zeros(N,1);
for i = 1:N
    samples = cell2mat(training_set(:,N));
    dist = pdist(samples, 'euclidean', 'Smallest');
    dist = squareform(dist);
    dist = dist(dist~=0);
    dist = reshape(dist, M-1, M);
    dist = min(dist);
    sigma_min_all(i) = mean(dist);
end

sigma_min = mean(sigma_min_all);
%disp(sigma_min);

sigma = T; % inital sigma
step = 1;  % sigma decrease step
t = 1;     % trial number
while (sigma > sigma_min) && (isfinite(alpha(t)))
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
                    difficult_data(end+1,:)=training_set(i,:);
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
            %fprintf("Training SVM %d out of %d\n", f, N);
            X = cell2mat(Sx(:,f));
            svm_model{f}=fitcsvm(X,Y,'KernelFunction','rbf', 'KernelScale', sigma);
            svm_labels(:,f)=predict(svm_model{f}, X);
        end 

    % 3. intermediate classifier h(t) = linear combination of svm classifiers
        h_model{t} = svm_model;
        h_weights(:,t) = calc_weights(svm_labels, Y, N, M, D);
        h_labels(:,t) = weighted_vote(svm_labels, h_weights(:,t));

    % 4. calculate classification weight for intermediate classifier h(t)
         alpha(t) = calc_weights(h_labels(:,t), Y, 1, M, D);
         fprintf("Alpha %d = %f.\n", t, alpha(t));
         if(~isfinite(alpha(t)))
             break;
         end       

    % 5. Update training sample weights
         D=D.*exp((-1).*Y.*alpha(t).*h_labels(t));
         D=D./sum(D);
    
    % 6. Update sigma and trial number
        sigma = sigma-step;
        t = t+1;
    toc
end

% set T = number of trials completed, and delete empty columns
 T = t-1; 
 alpha = alpha(1:T);
 h_weights = h_weights(:,1:T);
 h_model = h_model(1:T);

    
fprintf("Final votes for testing and training sets\n");
tic
H_labels = zeros(M,T);
for t = 1:T
    H_labels(:,t) = combo_predict(h_model{t}, training_set, N, h_weights(:,t));
end

ada_labels(:,1) = weighted_vote(H_labels, alpha);
ada_labels(ada_labels == -1) = 0;
toc
end

% Calculate weights for classifiers based on their classification error
function weights = calc_weights(trained_labels, class_labels, N, M, D)
err = zeros(1,N);
weights = zeros(1,N);
   for n = 1:N
        for m=1:M
            if (trained_labels(m,n)~=class_labels(m))
                err(n)=err(n)+D(m); 
            end  
        end
        weights(n)=0.5*log((1-err(n))/err(n));
   end

end
 
% Assign labels to data based on a weighted average of classifiers
function combo_labels = weighted_vote(trained_labels, weights)
    combo_labels(:,1) = sign(sum(trained_labels*weights,2));
    combo_labels(combo_labels==0) = 0;
end

% Produce classification output for a set of N weighted weak classifiers 
function predictions = combo_predict(models, testing_table, N, weights)
    weak_predictions = zeros(size(testing_table,1),N);
    for f = 1:N
        Xtest = cell2mat(testing_table(:,f));
        weak_predictions(:,f) = predict(models{f}, Xtest);
    end
    predictions = weighted_vote(weak_predictions, weights);
end