clear;
clc;

testdata_dir = fullfile('D:\Programming workspaces\Notepad workspace\CMCC projects python\Face blur detection\isblur');
dir_files = dir(fullfile(testdata_dir,'*.jpg'));
fileNames = {dir_files.name};
f = fopen('results.txt','w');
fprintf(f,'%7s %9s %9s\r\n','Name','unblured','BlurExtent');
for i = 1: length(fileNames)
    img = imread(fullfile(testdata_dir, fileNames{i}));


    if ndims(img) > 2
        img = rgb2gray(img);
    end

    % % gaussian blur
    % sigma = 0.01;
    % gfilter = fspecial('gaussian', [5 5], sigma);
    % img = imfilter(img, gfilter, 'replicate');
    % imshow(img);

    [unblured, BlurExtent] = blurDetection(img, 35, 0.05);
    
    fprintf(f,'%9s %3d %11.4f\r\n',fileNames{i}, unblured, BlurExtent);
end 
fclose(f);