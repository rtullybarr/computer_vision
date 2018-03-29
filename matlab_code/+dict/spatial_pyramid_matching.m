function single_descriptor = spatial_pyramid_matching(dictionary, feature_descriptors, lambda)
% SPATIAL_PYRAMID_MATCHING
    
    % feature descriptors: an RxCxd matrix of features
    % get responses of dictionary to feature descriptors
    [r, c, d] = size(feature_descriptors);
    
    whole_image_responses = dict.optimize_assignments(dictionary, reshape(feature_descriptors, [], d)', lambda);
    whole_image_responses = max(whole_image_responses, [], 2)';
    
    x_step = floor(r/2);
    y_step = floor(c/2);
    
    quarter_responses = cell(2);
    for i = 1:2
        for j = 1:2
            x_start = (i - 1)*x_step + 1;
            y_start = (j - 1)*y_step + 1;
            x_end = x_start + x_step - 1;
            y_end = y_start + y_step - 1;
            desc = feature_descriptors(x_start:x_end, y_start:y_end, :);
            resp = dict.optimize_assignments(dictionary, reshape(desc, [], d)', lambda);
            quarter_responses{i, j} = max(resp, [], 2)';
        end
    end
    
    x_step = floor(r/4);
    y_step = floor(c/4);
    sixteenth_responses = cell(4);
    for i = 1:4
        for j = 1:4
            x_start = (i - 1)*x_step + 1;
            y_start = (j - 1)*y_step + 1;
            x_end = x_start + x_step - 1;
            y_end = y_start + y_step - 1;
            desc = feature_descriptors(x_start:x_end, y_start:y_end, :);
            resp = dict.optimize_assignments(dictionary, reshape(desc, [], d)', lambda);
            sixteenth_responses{i, j} = max(resp, [], 2)';
        end
    end
    
    single_descriptor = [whole_image_responses, quarter_responses{:}, sixteenth_responses{:}];
end

