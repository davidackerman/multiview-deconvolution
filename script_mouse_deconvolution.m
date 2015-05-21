%%
%parameters

%imPath = 'E:\mouse_deconvolution\15-04-03\TM0010_v2\'
imPath = 'D:\Fernando\mouse_deconvolution\TM0010\'
imFilename = {'TP0_Channel0_Illum0_Angle0.tif', 'TP0_Channel0_Illum0_Angle90.tif', 'TP0_Channel0_Illum0_Angle180.tif','TP0_Channel0_Illum0_Angle270.tif'};%images are already registered in a common framework (Fiji Plugin withlinear interpolation

%outputBasename = 'E:/temp/deconvolution/mouse_fft_denoised'
outputBasename = 'D:/temp/deconvolution/mouse_fft_denoised'

%theoretical one
imPSF = {'transfomed PSF of viewsetup 0.tif', 'transfomed PSF of viewsetup 1.tif', 'transfomed PSF of viewsetup 2.tif', 'transfomed PSF of viewsetup 3.tif'};
%practical one
%imPSF = {'transfomed_PSF_experimental_mouse_angle0000.tif', 'transfomed_PSF_experimental_mouse_angle0090.tif', 'transfomed_PSF_experimental_mouse_angle0180.tif', 'transfomed_PSF_experimental_mouse_angle0270.tif'};


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
    weightsCell{ii} = single(estimateDeconvolutionWeights(imCell{ii}, 1, 15, []));
end

%%
%perform lucy richardson
numIters = 20;
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, numIters, lambdaTV, 0, outputBasename);