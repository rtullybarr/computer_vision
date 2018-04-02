wildebeest = preprocess(get_image_filenames('wildebeest', '*.jpg'), [256 256]);
guineaFowl = preprocess(get_image_filenames('guineaFowl', '*.jpg'), [256 256]);
hartebeest = preprocess(get_image_filenames('hartebeest', '*.jpg'), [256 256]);
giraffe = preprocess(get_image_filenames('giraffe', '*.jpg'), [256 256]);

species_names = ["wildebeest", "guineaFowl", "hartebeest", "giraffe"];
species_masks = [ones(length(wildebeest), 1); ones(length(guineaFowl), 1) .* 2; ones(length(hartebeest), 1) .* 3; ones(length(giraffe), 1) .* 4];

all_images = [wildebeest; guineaFowl; hartebeest; giraffe];

positive_class = 1;

class_labels = zeros(length(all_images), 1);
class_labels(species_masks == positive_class) = 1;
        
trained_model = cross_validate_ada(5, class_labels, LBP_image_vectors, SIFT_image_vectors);

        