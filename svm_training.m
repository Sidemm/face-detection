function svmClassifier = svm_training(features_pos, features_neg)
% This function will train your SVM classifier using the positive
% examples (features_pos) and negative examples (features_neg).

w = rand(size(features_pos,2),1); %placeholder, delete
b = rand(1); 


lambda = 0.00001;
     

total_features = [features_pos ; features_neg]; % positive and negative features concatanated vertically
[row,column] = size(total_features);
classification = ones (1,row); % filled all with ones at the beginning
classification(size(features_pos,1)+1 : end) = -1; % filled with -1s for negative features
[w b] = vl_svmtrain(total_features', classification', lambda) 

svmClassifier = struct('weights',w,'bias',b);
end