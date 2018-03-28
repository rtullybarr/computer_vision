function normalized_dictionary = normalize(dictionary)
%NORMALIZE: normalizes the input matrix 'dictionary'
    normalized_dictionary = dictionary ./ repmat(sqrt(sum(dictionary.^2)), [size(dictionary, 1), 1]);
end

