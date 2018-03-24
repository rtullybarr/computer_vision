function main(dictionary_size, dictionary_iterations, lambda)
    %Using LBP to train dictionary.
    fprintf("Reading and preprocessing images.");
    tic
    giraffes = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);
    toc

    fprintf("Extracting local binary patterns.");
    tic
    descriptors = cell(length(giraffes), 1);
    for i = 1:length(giraffes)
        descriptors{i} = LBP(giraffes{i});
    end
    toc

    fprintf("Selecting random subset of features for dictionary.");
    tic
    descriptors_flat = cell(length(descriptors), 1);
    for i = 1:length(descriptors)
        % desc: a RxCxd matrix of descriptors
        desc = descriptors{i};
        [~, ~, d] = size(desc);
        descriptors_flat{i} = reshape(desc, [], d);
        descriptors_flat{i} = descriptors_flat{i}';
    end

    all_descriptors = cat(2, descriptors_flat{:});

    % select 20% of the descriptors
    [~, num_descriptors] = size(all_descriptors);
    perm = randperm(num_descriptors);
    top = floor(num_descriptors*0.2);
    dictionary_learning_set = all_descriptors(:, perm(1:top));
    
    toc

    fprintf("Learning dictionary.");
    tic
    % learn the dictionary
    dictionary = learn_dictionary(dictionary_learning_set, dictionary_size, dictionary_iterations, lambda);
    toc
    
    fprintf("Assembling image vectors using SPM.");
    tic
    image_vectors = cell(length(descriptors));
    for i = 1:length(descriptors)
        % Use SPM to get a single image vector
        image_vectors{i} = spatial_pyramid_matching(dictionary, descriptors{1}, lambda);
    end
    toc
    
    % Associate image vectors with species label

    % 
end
