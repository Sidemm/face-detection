# face-detection
Hog feature creation and classification for face detection. 

In get_training_features file I read positive samples and changed their type to single before calling 
vl_hog function which calculated histogram of the images and created the features from samples. Then I 
added them to features_pos matrix and continued with negative samples. The number of samples 
required were greater than number of images so I gathered more than one sample from each image and 
calculated a variable sample_per_image by dividing them. Like in the positive samples I changed their type 
to single and cropped them randomly with the height and weight of the given parameters. This new 
samples was send to vl_hog function and added them to negative feature matrix. Since I took the ceiling 
of the division which determined sample per image, number of features exceeded the number of samples 
required. So I randomly deleted the featured to comply with this number.  
 In the svm_training file, in order to classify the HoG features I added them all to a new matrix. I 
created another one to label the data and filled with 1s at the beginning. Then I added -1s for negative features. 
Lastly I sent both matrices to vl_svmtrain to classify the data and calculate weights and bias. 
 
 In the next part I calculated the performance of the code by finding TP, FP, TN and FN values. For 
that I  created two for loops for positive and negative features. For positive features if the confidence was 
greater than the given threshold we knew that the there was a face and it was predicted so I increased 
the number of TP, otherwise increased FN. Likewise in the negative features if the confidence was greater 
than the given threshold we knew that there was no face and program detected a face so I increased the 
number of FP, otherwise decreased TN. At the end I calculated necessary performances according to 
instructions. The results were like the following; 
 Accuracy : 0. 9783, 0.9981, 0.9837,...... 
 Recall: 0.9625, 0.9852, 0.9810,....... 
 Tn Rate: 0.9889, 0.9914, 0.9843,........ 
 Precision: 0.9831, 0.9756, 0.9798,....... 
From these results I concluded that the classifier was frequently correct, it could succesfully detect faces 
when there is a face,  it did not give positive results when there isn’t a face almost never and when it gives 
positive results it’s extremely likely that there is a face.  
 
 In the last part I started converting the non gray images to grayscale after getting their color 
channel by size(image) function. If the channels are more than one it was a colored image. After that in 
the for loop which decreased the scale of window every time, I computed HoG features for each scaled 
window. In another for loop which enables windows to slide in both directions, I moved the window on 
the previously calculated HoG features. In addition I calculated the steps by dividing template size to hog 
cell size, because each cell was created with that many pixels (6x6). Therefore windows were moving 6 
pixels in each step. After that I calculated the confidence and also found their original x and y locations if 
the confidence were above a certain threshold. When the threshold was 0.9 the average accuracy was 
0.83 , with 0.75 again 0.83, lastly I tried with 0.8 and it increased to 0.85 so I left the threshold as 0.8. 
 
