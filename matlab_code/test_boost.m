class1 = {'test_images/monarch1.jpg';'test_images/monarch2.jpg';'test_images/monarch3.jpg'};
class2 = {'test_images/moth2.jpg'; 'test_images/moth1.jpg'; 'test_images/tulips.png'};

class1 = preprocess(class1, [256, 256]);
class2 = preprocess(class2, [256, 256]);

class1_LBP = [feats.LBP(class1{1}); feats.LBP(class1{2}); feats.LBP(class1{3})];
class1_HOG = [feats.HOG(class1{1}); feats.HOG(class1{2}); feats.HOG(class1{3})];

class1_features = [class1_LBP class1_HOG];

class2_LBP = [feats.LBP(class2{1}); feats.LBP(class2{2}); feats.LBP(class2{3})];
class2_HOG = [feats.HOG(class2{1}); feats.HOG(class2{2}); feats.HOG(class2{3})];

class2_features = [class2_LBP class2_HOG];

[class1Samps, idx1] = datasample(class1_features(1:256,:), 50);
[class2Samps, idx2] = datasample(class2_features(1:256,:), 50); 

class_labels = [ones(50,1); zeros(50,1)];

boosted_features = combine_features([class1Samps; class2Samps], class_labels);
[pX,pIdx] = datasample(class1_features(257:512, :),6);
[labels, score] = predict(boosted_features,pX);
table(label,pIdx','VariableNames',{'Predicted', 'Index'})
for i = 1:6
    subplot(1,6,i)
    imshow(mcells{pIdx(i)})
end