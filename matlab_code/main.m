% test out using LBP to train dictionary.
giraffes = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);

descriptors = cell(length(giraffes), 1);
for i = 1:length(giraffes)
    descriptors{i} = LBP(giraffes{i});
end

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
%size(dictionary_learning_set)

% learn the dictionary
dictionary = learn_dictionary(dictionary_learning_set, 256, 25);

% get response to one image
response = optimize_assignments(dictionary, descriptors_flat{1}, 0.026);

% spm
image_vector = spatial_pyramid_matching(dictionary, descriptors{1}, 0.026);

