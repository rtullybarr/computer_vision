function features = HOG(img)
    features = extractHOGFeatures(img, 'CellSize', [4,4], 'BlockSize', [4,4], 'BlockOverlap', [0,0], 'NumBins', 8);
    features = reshape(features, [128, length(features)/128]);
    features = double(features');
end
