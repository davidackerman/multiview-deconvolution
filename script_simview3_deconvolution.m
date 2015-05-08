

imPath = 'E:\simview3_deconvolution\15_04_24\TM1445_2views\BMx3Ddenoised\'
imFilename = {'TP0_Channel0_Illum0_Angle0_coarselyAligned','TP0_Channel0_Illum0_Angle1_coarselyAligned_registered'};

lambdaTV = 0.008; %0.002 is value recommended by paper

%%
%load images
numImg = length(imFilename);
imCell = cell(numImg,1);

for ii = 1:numImg
   imCell{ii} = readTIFFstack([imPath imFilename{ii} '.tif']); 
end

%%
%{
disp '==================CROPPING DATASET========================'
for ii = 1:numImg
    imCell{ii} = imCell{ii}(:,1:610,:);
end
%}

%%
%generate PSF
sampling = 0.40625 * ones(1,3);%in um
FWHMpsf = [0.6 0.6 3.0];%in um
PSFcell = cell(numImg,1);
for ii = 1:numImg    
    
    filenameOut = [imPath 'PSF_view' num2str(ii) '.raw'];
    PSFcell{ii} = generatePSF(sampling,FWHMpsf, filenameOut);
    
    FWHMpsf = circshift(FWHMpsf,[0 1]);
end

%%
%perform lucy richardson
numIters = 20;
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, numIters, lambdaTV, 0, 'E:/temp/deconvolution/simview3_fft_denoised');