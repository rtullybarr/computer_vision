function images = read_images(directory)
%READ_IMAGES Reads in all images in the given directory and returns them.
    filenames = dir(fullfile(directory, '*.jpg'));
    images = cell(length(filenames), 1);
    
    for i = 1:length(filenames)
        images{i} = imread(fullfile(filenames(i).folder, filenames(i).name));
    end
end

