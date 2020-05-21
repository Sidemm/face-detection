function [accuracy,recall,tn_rate,precision] = classifier_performance(svmClassifier,features_pos,features_neg)

% This function will return some metrics generally computed to determine the
% performance of a binary classifier. To clasify a sample first determine 
% the confidence value (also referred to as score): 
% confidence = features*w +b, 
% where features are the hog features, w is the linear classifier 
% weights and b is the classifier bias. 

accuracy=0; recall=0; tn_rate=0; precision=0;

 TP = 0; FP = 0; TN = 0; FN = 0;
 for i = 1 : size(features_pos,1) % true positives 
    confidence = features_pos(i,:) *svmClassifier.weights + svmClassifier.bias;
    if confidence >=0 % result of the classifier
    TP = TP + 1; % if positive 
    else 
    FN = FN + 1; % if negative
    
    end
    
 end
 
 for i = 1 : size(features_neg,1) % true negatives
    confidence = features_neg(i,:) *svmClassifier.weights + svmClassifier.bias;
    if confidence >=0 % result of the classifier
    FP = FP + 1; % if positive 
    else 
    TN = TN + 1; % if negative
    
    end
    
 end
accuracy  = (TP+TN)/(TP+TN+FP+FN);
recall  = TP/(TP+FN);
tn_rate  = TN/(TN+FP);
precision  = TP/(TP+FP);


end