

imPath = 'S:\SiMView3\15-05-19\Beads\deconvolution\'
imFilename = {'TP0_Channel0_Illum0_Angle0', 'TP0_Channel0_Illum0_Angle1', 'TP0_Channel0_Illum0_Angle2', 'TP0_Channel0_Illum0_Angle3'};
psfFilename = {'transfomed PSF of viewsetup 0', 'transfomed PSF of viewsetup 1', 'transfomed PSF of viewsetup 2', 'transfomed PSF of viewsetup 3'};

lambdaTV = 0.008; %0.002 is value recommended by paper
numIters = 40;
backgroundOffset = 100;

%%
%load images
numImg = length(imFilename);
imCell = cell(numImg,1);
imPSF = cell(numImg,1);
for ii = 1:numImg
   imCell{ii} = readTIFFstack([imPath imFilename{ii} '.tif']); 
   imPSF{ii}  = readTIFFstack([imPath psfFilename{ii} '.tif']); 
end


%%
%perform lucy richardson

J = multiviewDeconvolutionLucyRichardson(imCell,imPSF, [], backgroundOffset, numIters, lambdaTV, 0, [imPath 'LRdeconvolution_iter']);
