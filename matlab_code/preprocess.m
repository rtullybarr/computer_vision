function images = preprocess(images)
%PREPROCESS - perform preprocessing on the input images
% currently, just converts them to grayscale
    for i = 1:length(images)
        images{i} = rgb2gray(images{i});
    end
end

