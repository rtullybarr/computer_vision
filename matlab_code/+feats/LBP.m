% This function takes an image and outputs cell-structured LBP features 
% INPUT: 
%   Required: img -> greyscale image, RxC
%   Optional: 
%   1. cellsize: 2x2 array, size of cells, default = [16,16]
%   2. filt: parameter for filtering, default = 0 (no filtering)
%   3. rot: whether to implement rotation invariance default = 0 (false)
% OUTPUT: (R/cellsize)x(C/cellsize)x59 matrix of LBP features (or RxCx10 for rot. invariant)
%         Change to 2D matrix: features = reshape(features, [RxC, 59]) 


function features = LBP(img, varargin)
    Defaults = {[16,16], 0, 0};
    idx = ~cellfun('isempty',varargin);
    Defaults(idx) = varargin(idx);

    cellsize = Defaults{1};
    filt = Defaults{2};
    rot = Defaults{3};

    s = size(img);
    r = floor(s(1)/cellsize(1));
    c = floor(s(2)/cellsize(2));
    
    % 1. Pre-processing: edge-preserving noise removal filter
    if(filt==1)
        img_smooth = imgaussfilt(img,3);
        img = img-img_smooth;
    end
    
    % 2. Exttract LBP in cells ( e.g. 59-length histogram for each 16x16 cell)
    
    % No rotational invariance, 59 bins for each cell
    if(rot == 0) 
        features = extractLBPFeatures(img,'CellSize', cellsize);
        features = reshape(features, [59, length(features)/59]);
        features = features';
        features = reshape(features, [r, c, 59]);
   
    % Make LBP features rotationally invariant, 10 bins for each cell
    else 
        features = extractLBPFeatures(img,'CellSize', cellsize, 'Upright', false);
        features = reshape(features, [legnth(features)/10, 10]);
        features = features';
        features = reshape(features, [r, c, 10]);
    end
    
    % 3. tri-linear interpolation?
    
    % Convert to double
    features = double(features);
end
