%%
%parameters

imPath = 'E:\mouse_deconvolution\15-04-03\TM0010_v2\'
imFilename = {'TP0_Channel0_Illum0_Angle0.tif', 'TP0_Channel0_Illum0_Angle90.tif', 'TP0_Channel0_Illum0_Angle180.tif','TP0_Channel0_Illum0_Angle270.tif'};%images are already registered in a common framework (Fiji Plugin withlinear interpolation

imPSF = {'transfomed PSF of viewsetup 0.tif', 'transfomed PSF of viewsetup 1.tif', 'transfomed PSF of viewsetup 2.tif', 'transfomed PSF of viewsetup 3.tif'};

lambdaTV = 0.008; %0.002 is value recommended by paper

%%
%load images and PSF
numImg = length(imFilename);
imCell = cell(numImg,1);
PSFcell = cell(numImg,1);
for ii = 1:numImg
   imCell{ii} = readTIFFstack([imPath imFilename{ii}]);
   PSFcell{ii} = readTIFFstack([imPath imPSF{ii}]);
end

%%
%calculate weights based DCT Shannon entropy contrast
weightsCell = cell(numImg,1);
for ii = 1:numImg
    weightsCell{ii} = estimateDeconvolutionWeights(imCell{ii}, 1, 15, []);
end

%%
%perform lucy richardson
numIters = 20;
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, numIters, lambdaTV, 0, 'E:/temp/deconvolution/simview3_fft_denoised');