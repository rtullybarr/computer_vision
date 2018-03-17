% Local Structured descriptor implementation
% INPUT: RxCxZ matrix of histogram features 
%   R = # rows of cells
%   C = # columns of cells
%   Z = # features in each histogram
% OUTPUT: RxCx7 matrix of local-structured descriptors

function LS_features = LS_descriptor(features)
    h = 0.4743; % normalization factor for Local structures
    h1 = 0.1; % normalization factor for local overall structure
    
    r = size(features,1);
    c = size(features,2);
    
    LS_features = zeros(r-2,c-2,7);
    LR = zeros(4,1);

    for i = 2:r-1
        for j = 2:c-1
        H = sum(features(i,j,:),3);
        E = sum((features(i-1:i+1,j-1:j+1,:).^2),3);
        
        LR(1) = H/sqrt(E(1,1)+E(1,2)+E(2,1)+E(2,2));
        LR(2) = H/sqrt(E(1,2)+E(1,3)+E(2,2)+E(2,3));
        LR(3) = H/sqrt(E(2,1)+E(2,2)+E(3,1)+E(3,2));
        LR(4) = H/sqrt(E(2,2)+E(2,3)+E(3,2)+E(3,3));
        
        LHS1 = h*abs(LR(1)-LR(2));
        LHS2 = h*abs(LR(3)-LR(4));
        LVS1 = h*abs(LR(1)-LR(3));
        LVS2 = h*abs(LR(2)-LR(4));
        LDS1 = h*abs(LR(1)-LR(4));
        LDS2 = h*abs(LR(2)-LR(3));
        LOS = h1*abs(sum(LR));
        
        LS_features(i-1,j-1,:) = [LHS1, LHS2, LVS1, LVS2, LDS1, LDS2, LOS];
        end
    end        
end