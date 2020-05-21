function [bboxes, confidences, image_ids] = ....
    run_detector(test_data_path, svmClassifier, hog_template_size,hog_cell_size)

% This function returns detections on all of the images in 'test_data_path'.

test_scenes = dir( fullfile( test_data_path, '*.jpg' ));
num_images = length(test_scenes);

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

for i = 1:num_images
    
    image_name = test_scenes(i).name;
    fprintf('Detecting faces in %s\n', image_name);
    img = imread( fullfile( test_data_path, test_scenes(i).name ));
    
    [cur_confidences,cur_bboxes] =...
        PlaceHolder(img, svmClassifier, hog_template_size,hog_cell_size);
    
    [cur_confidences,cur_bboxes] =...
        Detector(img, svmClassifier, hog_template_size,hog_cell_size);
    
    cur_image_ids = cell(0,1);
    cur_image_ids(1:size(cur_bboxes,1)) = {test_scenes(i).name}; 
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end
end



function [cur_confidences,cur_bboxes] = ...
    PlaceHolder( img,svmClassifier, hog_template_size,hog_cell_size)


% Creating 15 random detections per image
cur_x_min = rand(15,1) * size(img,2);
cur_y_min = rand(15,1) * size(img,1);
cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
cur_confidences = rand(15,1) * 4 - 2; %confidences in the range [-2 2]


[cur_bboxes, cur_confidences] = ...
    nonMaximum_Suppression(cur_bboxes, cur_confidences,size(img));
end


function [bboxes, confidences] = ...
    nonMaximum_Suppression(bboxes, confidences,img_size)

[is_maximum] = non_max_supr_bbox(bboxes, confidences, img_size);
confidences = confidences(is_maximum,:);
bboxes      = bboxes(is_maximum,:);
end


function [cur_confidences,cur_bboxes] = ...
    Detector(img, svmClassifier, hog_template_size,hog_cell_size)
 

cur_bboxes = zeros(0,4);
cur_confidences = zeros(0,1);
,
img = im2single(img);
[row columns channels] = size(img);
if channels > 1 % colored images
    img = rgb2gray(img)
end

for scale = 1 : -0.1 : 0.1 % different scaled windows
scaled_img = imresize(img,scale);
hog = vl_hog(scaled_img,hog_cell_size);
img_row = size (hog, 1);
img_column = size (hog, 2);
step = hog_template_size/hog_cell_size; % sliding step in pixels
    for i = 1 : img_row - step + 1
        for j = 1 : img_column - step + 1 
            window = hog(i:i+step-1, j:j+step-1,:); % hog features of window
            confidence = window(:)'*svmClassifier.weights + svmClassifier.bias;
            if confidence > 0.8 % finds the original locations for the detection 
            orj_xmin = (j*hog_cell_size - hog_cell_size + 1)/scale;
            orj_xmax = (j+step -1)*hog_cell_size/scale;
            orj_ymin = (i*hog_cell_size - hog_cell_size + 1)/scale;
            orj_ymax = (i+step -1)*hog_cell_size/scale;
            cur_bboxes = [cur_bboxes ; [orj_xmin,orj_ymin,orj_xmax,orj_ymax]]; % adds newly calculated bboxes
            cur_confidences = [cur_confidences ; confidence];% adds newly calculated confidences
            end
        end
    end
end

[cur_bboxes, cur_confidences] = ...
    nonMaximum_Suppression(cur_bboxes, cur_confidences,size(img));



end


