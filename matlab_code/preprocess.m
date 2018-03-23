function images = preprocess(images, imgsize)
%PREPROCESS - perform preprocessing on the input images
% Input: cell array of filenames, desired image size eg [256, 256]
% output: cell array of square greyscale images
    for i = 1:length(images)
        images{i} = imread(images{i});
        images{i} = rgb2gray(images{i});
        [y,x] = size(images{i});
        len = min(x,y);
        images{i} = imcrop(images{i}, [(x-len)/2, (y/len)/2, len, len]);
        images{i} = imresize(images{i}, imgsize);
    end
end

