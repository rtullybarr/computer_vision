function siftFeatures = sift_features(img)
% INPUT
% img - gray level image
%
% OUTPUT
% siftFeatures - sift feature vector of length r/4-3 x c/4-3 x 128
%                where r and c are size of img

% Note: 
% SIFT usually works with Gaussian smoothed images, but 
% the paper just mentioned converting images to grayscale before
% applying SIFT. Since they use SIFT and HOG interchangeably
% I'm assuming they used HOG with SIFT dimensions, rather than 
% closely following a typical dense SIFT implementation.

% SIFT Parameters
cellsize = [4 4];       % 4x4 pixel cells
blocksize = [4 4];      % 4x4 cell blocks
numbins = 8;            % 8 orientations for histo
stepsize = [0 0];       % 4 pixel step size (from paper) - corresponds to 3 overlapping blocks per feature calc

% SIFT Feature Dimensions
[r, c] = size(img);
r = floor(r/16);
c = floor(c/16);
% r = r/4 - 3;    % 3 = cellsize(1) - 1
% c = c/4 - 3;    % 3 = cellsize(2) - 1

[siftFeatures, visualization] = extractHOGFeatures(img,'CellSize',cellsize, 'BlockSize', blocksize,...
   'NumBins', numbins, 'BlockOverlap', stepsize, 'UseSignedOrientation', true);
siftFeatures = reshape(siftFeatures, [128, length(siftFeatures)/128]);
siftFeatures = siftFeatures';
siftFeatures = reshape(siftFeatures, [r, c, 128]);

% subplot(1,2,1);
% imshow(img);
% subplot(1,2,2);
% plot(visualization);

end