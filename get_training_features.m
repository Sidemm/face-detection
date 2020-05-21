function [features_pos, features_neg] = get_training_features...
    (train_path_pos, train_path_neg,hog_template_size,hog_cell_size)

%This function returns the hog features of all positive examples and for
%negative examples


image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);

features_pos = rand(num_images, (hog_template_size / hog_cell_size)^2 * 31);


for i = 1 : num_images
    img = image_files(i);
    img_path = fullfile(train_path_pos, img.name);
    image = imread(img_path); 
    image = im2single(image); % type is changed to single
    hog = vl_hog(image,hog_cell_size);
    features_pos(i,:) =  hog(:)';    
end    



sample_per_img = ceil(num_samples/num_images) %to reach the number of samples with the given images 
for i = 1 : num_images
    img = image_files(i);
    img_path = fullfile(train_path_neg, img.name);
    image = rgb2gray(imread(img_path));
    image = im2single(image); % type is changed to single
    [row, column] = size(image);
    for j = 1 : sample_per_img
    crop_row = randi(row-hog_template_size); % randomly chosen location to take samples
    crop_column = randi(column-hog_template_size);
    sample = image (crop_row:crop_row+hog_template_size-1,crop_column:crop_column+hog_template_size-1);
    hog = vl_hog(sample,hog_cell_size);
    features_neg(sample_per_img*i-sample_per_img+j,:) =  hog(:)';    
    end
   
end    
[features_row, features_column] = size(features_neg);

for i = 1 : features_row - num_samples % extra samples are deleted here
    row_to_delete = randi(features_row-i-1); % randomly chosen sample
    features_neg(row_to_delete,:) = [];
end



end

