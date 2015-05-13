%Downsample image by 2^numLevels using a Gaussian smoothing forantialiasing
function im = stackDownsample(im, numLevels)


for kk=1:numLevels
    im = imgaussian(im,sqrt(2.5));
    %downsample XYZ
    im = im(1:2:end,1:2:end,1:2:end);          
end