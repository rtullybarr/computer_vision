function images = get_image_filenames(directory, extension)
%GET_IMAGE_FILENAMES - returns a cell array containing filenames of all
%files with the given extension in the given directory.
    filenames = dir(fullfile(directory, extension));
    images = cell(length(filenames), 1);
    
    for i = 1:length(filenames)
        images{i} = fullfile(filenames(i).folder, filenames(i).name);
    end
end

