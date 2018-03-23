% test out using LBP to train dictionary.
giraffes = read_images('giraffe');
giraffes = preprocess(giraffes);

descriptors = cell(length(giraffes), 1);
for i = 1:length(giraffes)
    descriptors{i} = LBP(giraffes{i});
    %descriptors{i} = LS_descriptor(features);
end

% first descriptor
%descriptors{1}(1, 1, :)

% get random subset of descriptors
descriptors_flat = cell(length(descriptors), 1);
for i = 1:length(descriptors)
    % descriptor: a RxCx7 matrix of descriptors
    desc = descriptors{i};
    descriptors_flat{i} = reshape(desc, [], 59);
end

all_descriptors = cat(1, descriptors_flat{:});

% select 20% of the descriptors
perm = randperm(length(all_descriptors));
top = floor(length(all_descriptors)*0.2);
dictionary_learning_set = all_descriptors(perm(1:top), :);

% learn the dictionary
dictionary = learn_dictionary(dictionary_learning_set, 256, 25);

% get response to one image
response = optimize_assignments(dictionary, descriptors_flat{i});

% spm
image_vector = spatial_pyramid_matching(dictionary, descriptors{i});

