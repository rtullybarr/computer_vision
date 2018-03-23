function single_descriptor = spatial_pyramid_matching(dictionary, feature_descriptors)
% SPATIAL_PYRAMID_MATCHING
    
    % feature descriptors: an RxCx7 matrix of features
    % get responses of dictionary to feature descriptors
    whole_image_responses = optimize_assignments(dictionary, reshape(feature_descriptors, [], 7));
    whole_image_responses(find(whole_image_responses))
    size(whole_image_responses)
    whole_image_responses = max(whole_image_responses);
    whole_image_responses(find(whole_image_responses))
    [r, c, ~] = size(feature_descriptors);
    
    x_step = floor(r/2);
    y_step = floor(c/2);
    
    quarter_responses = cell(2);
    for i = 1:2
        for j = 1:2
            x_start = (i - 1)*x_step + 1;
            y_start = (j - 1)*y_step + 1;
            x_end = x_start + x_step;
            y_end = y_start + y_step;
            desc = feature_descriptors(x_start:x_end, y_start:y_end, :);
            quarter_responses{i, j} = max(optimize_assignments(dictionary, reshape(desc, [], 7)));
        end
    end
    
    x_step = floor(r/4);
    y_step = floor(c/4);
    sixteenth_responses = cell(4);
    for i = 1:4
        for j = 1:4
            x_start = (i - 1)*x_step + 1;
            y_start = (j - 1)*y_step + 1;
            x_end = x_start + x_step;
            y_end = y_start + y_step;
            desc = feature_descriptors(x_start:x_end, y_start:y_end, :);
            sixteenth_responses{i, j} = max(optimize_assignments(dictionary, reshape(desc, [], 7)));
        end
    end
    
    single_descriptor = [whole_image_responses, quarter_responses{:}, sixteenth_responses{:}];
end

