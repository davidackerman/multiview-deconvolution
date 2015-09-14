TM = 720
isMembrCh = true;

%%
TMstr = num2str(TM,'%.6d');
if( isMembrCh == true )
    imPath = ['\\dm11\kellerlab\deconvolutionDatasets\15_06_11_fly_functionalImage_TM' TMstr 'membrCh_Fiji\']
else
    imPath = ['\\dm11\kellerlab\deconvolutionDatasets\15_06_11_fly_functionalImage_TM' TMstr '_Fiji\']
end

imFilename = {'TP0_Channel0_Illum0_Angle0', 'TP0_Channel0_Illum0_Angle1', 'TP0_Channel0_Illum0_Angle2', 'TP0_Channel0_Illum0_Angle3'};
psfFilename = {'transfomed PSF of viewsetup 0', 'transfomed PSF of viewsetup 1', 'transfomed PSF of viewsetup 2', 'transfomed PSF of viewsetup 3'};


lambdaTV = 0.008; %0.002 is value recommended by paper
numIters = 100;
backgroundOffset = 100;

%%
%load images
numImg = length(imFilename);
imCell = cell(numImg,1);
weightsCell = cell(numImg,1);
imPSF = cell(numImg,1);
for ii = 1:numImg
    imCell{ii} = readTIFFstack([imPath imFilename{ii} '.tif']);
    imPSF{ii}  = readTIFFstack([imPath psfFilename{ii} '.tif']);
    
    
    %calculate contrast weights
    wFilename = [imPath 'simview3_TM' num2str(TM,'%6d') '_weightsReg_' num2str(ii) '.klb'];
    if( exist(wFilename) == 0 )
        disp('Calculating contrast weights');
        weightsCell{ii} = single(estimateDeconvolutionWeights(imCell{ii}, 1 , 15, []));%anisotropy here is 1 since they are already interpolated
        %save weights
        writeKLBstack(weightsCell{ii},wFilename);
    else
        weightsCell{ii} = readKLBstack(wFilename);
    end
    
    
end


%%
%perform lucy richardson
J = multiviewDeconvolutionLucyRichardson(imCell,imPSF, weightsCell, backgroundOffset, numIters, lambdaTV, 0, [imPath 'LRdeconvolution_iter']);
