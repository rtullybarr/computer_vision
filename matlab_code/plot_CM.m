function CM_labelled = plot_CM(CM)
% Takes a 4x4 confusion matrix and displays a color coded and labelled
% image. 
% **NOTE: For a good quality image you have to maximize the figure. Otherwise
% the text in the image doesn't show up well
%
% *Note: This could take larger confusion matrices, but the labelling would
% have to be changed.
%
    
% Initialize variables
cmap = ones(50,3);
b = [230, 249, 255];

% Create color map
incrementr = 20;
incrementg = 15;

% Light blue for lower values
for i = 2:12
    b(1) = b(1) - incrementr;
    b(2) = b(2) - incrementg;

    cmap(i, :) = b./255;
end

% Set middle values to gray (unused values)
gray = [194, 214, 214]./255;
cmap(13:32, :) = zeros(20,3);
cmap(13:32, :) = cmap(13:32, :) + gray;

% Dark blue for higher values
b = [0, 195, 255];
incrementg = 6;
incrementr = 7;
    
for i = 33:50
    b(2) = b(2) - incrementg;
    b(3) = b(3) - incrementr;

    cmap(i, :) = b./255; 
end

% Normalize confusion matrix
row_sum = sum(CM, 2);
CM_norm = CM./row_sum;

% Resize image so it's large enough to add text
CM_img_resized = imresize(CM_norm,  [512 512], 'nearest');

% Display values from confusion matrix
[r_img, c_img] = size(CM_img_resized);
[r_cm, c_cm] = size(CM_norm);

positions = ones(r_cm*c_cm, 2);
values = cell(r_cm*c_cm, 1);
txtColor = zeros(r_cm*c_cm, 3);
tix = zeros(1,4);

r_increment = r_img/r_cm;
c_increment = c_img/c_cm;
ind = 1;

for i = 1:r_cm
    for j = 1:c_cm
        positions(ind,1) = r_increment*(i-0.5);
        positions(ind,2) = c_increment*(j-0.5);
        
        % Round to 3 decimal places
        values{ind} = num2str(round(CM_norm(i,j)*1000)/1000);
        
        if i == j
            txtColor(ind, :) = [1 1 1];
        end
        
        ind = ind + 1;
    end
    tix(i) = r_increment*(i-0.5);
end

% Convert image with scaled colors to RGB
CM_img_resized = round(CM_img_resized*100);
CM_img_rgb = ind2rgb(CM_img_resized,cmap);

% Insert confusion matrix values into figure
CM_labelled = insertText(CM_img_rgb, positions, values,...
    'AnchorPoint', 'Center' ,'TextColor',txtColor,'BoxColor', 'white', 'FontSize',14,'BoxOpacity',0);

figure
imshow(CM_labelled)
title('Confusion Matrix')
ylabel('Actual')
xlabel('Predicted')
set(gca, 'visible', 'on')
set(gca,'Position',[0 0.05 1 .8]);
%set(gca,'OuterPosition',[0 0 .99 .99]);  

xticks(tix)
yticks(tix)

labels = {'1', '2', '3', '4'};
xticklabels(labels)
yticklabels(labels)
set(gca,'xaxisLocation','top')

end


