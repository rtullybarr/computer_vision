% reproducibility
rng(1);

%dictionary_sizes = [128, 256, 512];
dictionary_size = 128;
%dictionary_iterations = [10, 20, 30];
dictionary_iterations = 10;
lambdas = [0.018, 0.022, 0.026, 0.03, 0.034];
training_set_size = 20000;

fprintf("Reading and preprocessing images.\n");
tic
wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
guineaFowl = preprocess(get_image_filenames('guineaFowl', '*.jpg'), [256 256]);
hartebeest = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
giraffe = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);

species_names = ["wildebeest", "guineaFowl", "hartebeest", "giraffe"];
species_masks = [ones(length(wildebeest), 1); ones(length(guineaFowl), 1) .* 2; ones(length(hartebeest), 1) .* 3; ones(length(giraffe), 1) .* 4];

all_images = [wildebeest; guineaFowl; hartebeest; giraffe];

for l = 1:length(lambdas)
    lambda = lambdas(l);
    fprintf("Extracting local binary patterns.\n");
    tic
    LBP_features = cell(length(all_images), 1);
    for i = 1:length(all_images)
        LBP_features{i} = feats.LBP(all_images{i});
    end
    toc

    fprintf("Learning dictionary - LBP.\n");
    tic
    LBP_dictionary = dict.learn_dictionary(LBP_features, training_set_size, dictionary_size, dictionary_iterations, lambda);
    toc

    fprintf("Extracting SIFT descriptors.\n");
    tic
    SIFT_features = cell(length(all_images), 1);
    for i = 1:length(all_images)
        SIFT_features{i} = double(feats.sift_features(all_images{i}));
    end
    toc

    fprintf("Learning dictionary - SIFT.\n");
    tic
    SIFT_dictionary = dict.learn_dictionary(SIFT_features, training_set_size, dictionary_size, dictionary_iterations, lambda);
    toc

    fprintf("Assembling image vectors using SPM.\n");
    tic

    % known quantity: the length of the image_vector returned by the SPM
    % step is dict_size + 4*dict_size + 16*dict_size
    img_vector_len = 21*dictionary_size;

    LBP_image_vectors = zeros(length(LBP_features), img_vector_len);
    for i = 1:length(LBP_features)
        % Use SPM to get a single image vector
        LBP_image_vectors(i, :) = dict.spatial_pyramid_matching(LBP_dictionary, LBP_features{i}, lambda);
    end

    SIFT_image_vectors = zeros(length(SIFT_features), img_vector_len);
    for i = 1:length(SIFT_features)
        % Use SPM to get a single image vector
        SIFT_image_vectors(i, :) = dict.spatial_pyramid_matching(SIFT_dictionary, SIFT_features{i}, lambda);
    end

    toc


    suffix = [char(species_names(positive_class)) '_dictsize_' num2str(dictionary_size) '_iter_' num2str(dictionary_iterations) '_lambda_' num2str(lambda) * 100];

    save(['LPB_dict_' suffix], 'LBP_dictionary');
    save(['LPB_img_vec_' suffix], 'LBP_image_vectors');
    save(['SIFT_dict_' suffix], 'SIFT_dictionary');
    save(['SIFT_img_vec_' suffix], 'SIFT_image_vectors');
    toc

end
