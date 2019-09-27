function [unblured, BlurExtent] = blurDetection(img, threshold, MinZero)

% 2004 Blur Detection for Digital Images Using Wavelet Transform

if nargin < 3
    MinZero = 0.05;
    if nargin < 2
        threshold = 35;
        if nargin < 1
            img = imread('lena.png');
        end
    end
end

if ndims(img) > 2
    img = rgb2gray(img);
end

img0 = single(img);

[m0, n0] = size(img);
m = ceil(single(m0) / 16) * 16;
n = ceil(single(n0) / 16) * 16;
img = zeros(m, n);
img(1:m0, 1:n0) = img0;

tic;

%% Algorithm 1: HWT for edge detection
% Step1 (Harr wavelet transform)
% level1
[m, n] = size(img);
level1 = getWaveletLevel(img);
% level2
m = m / 2; n = n / 2;
level2 = getWaveletLevel(level1(1:m, 1:n));
% level3
m = m / 2; n = n / 2;
level3 = getWaveletLevel(level2(1:m, 1:n));

% Step2
[m, n] = size(img);
Emap1 = sqrt(level1(1:m/2, n/2+1:n).^2 + level1(m/2+1:m, 1:n/2).^2 + level1(m/2+1:m, n/2+1:n).^2);
m = m/2; n = n/2;
Emap2 = sqrt(level2(1:m/2, n/2+1:n).^2 + level2(m/2+1:m, 1:n/2).^2 + level2(m/2+1:m, n/2+1:n).^2);
m = m/2; n = n/2;
Emap3 = sqrt(level3(1:m/2, n/2+1:n).^2 + level3(m/2+1:m, 1:n/2).^2 + level3(m/2+1:m, n/2+1:n).^2);
% Step3
Emax1 = getEmax(Emap1, 8);
Emax2 = getEmax(Emap2, 4);
Emax3 = getEmax(Emap3, 2);

%% Algorithm2: blur detection scheme
% Step1 (Alegorithm 1)

% Step2 (Rule1)
[m, n] = size(Emax1);
Nedge = 0;
Eedge = zeros(m, n);
for i = 1:m
    for j = 1:n
        if Emax1(i, j) > threshold || Emax2(i, j) > threshold || Emax3(i, j) > threshold
            Nedge = Nedge + 1;
            Eedge(i, j) = 1;
        end
    end
end
% Step3 (Rule2)
Nda = 0;
for i = 1:m
    for j = 1:n
        if Eedge(i, j) == 1 ...
                && Emax1(i, j) > Emax2(i, j) && Emax2(i, j) > Emax3(i, j)
            Nda = Nda + 1;
        end
    end
end
% Step4 (Rule3,4)
Nrg = 0;
Eedge_Gstep_Roof = zeros(m, n);
for i = 1:m
    for j = 1:n
        if Eedge(i, j) == 1 ...
                && (Emax1(i, j) < Emax2(i, j) && Emax2(i, j) < Emax3(i, j) ...
                    || (Emax2(i,j) > Emax1(i,j) && Emax2(i,j) > Emax3(i,j)))
            Nrg = Nrg + 1;
            Eedge_Gstep_Roof(i, j) = 1;
        end
    end
end
% Step5 (Rule5)
Nbrg = 0;
for i = 1:m
    for j = 1:n
        if Eedge_Gstep_Roof(i, j) == 1 && Emax1(i, j) < threshold
            Nbrg = Nbrg + 1;
        end
    end
end
% Step6
Per = double(Nda) / Nedge;
unblured = Per > MinZero;

% Step7
BlurExtent = double(Nbrg) / Nrg;

toc;


%
function [ level1 ] = getWaveletLevel( img )
[m, n] = size(img);
% haar wavelet trainform
level1_horizontal = zeros(m, n);
for i = 1:n/2
    level1_horizontal(:, i) = (img(:, 2*i-1) + img(:, 2*i)) / 2;
    level1_horizontal(:, i+n/2) = img(:, 2*i-1) - level1_horizontal(:, i);
end
level1 = zeros(m, n);
for i = 1:m/2
    level1(i, :) = (level1_horizontal(2*i-1, :) + level1_horizontal(2*i, :)) / 2;
    level1(i+m/2, :) = level1_horizontal(2*i-1, :) - level1(i, :);
end


function Emax = getEmax(Emap, scale)
[m, n] = size(Emap);
Emax = zeros(m/scale, n/scale);
for i = 1:m/scale
    for j = 1:n/scale
        Emax(i,j) = max(max(Emap(scale*(i-1)+1:scale*i, scale*(j-1)+1:scale*j)));
    end
end