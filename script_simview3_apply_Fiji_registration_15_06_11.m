TM = 1
isMembrCh = false;

%%
TMstr = num2str(TM,'%.6d');
imPath = ['S:\SiMView3\15-06-11\Dme_E2_His2AvRFP_spiderGFP_12-03_20150611_155054.corrected\SPM00\TM' TMstr '\'];
if( isMembrCh == true )
    imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN03.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN03.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees
    outputFolder = ['\\dm11\kellerlab\deconvolutionDatasets\15_06_11_fly_functionalImage_TM' TMstr 'membrCh_Fiji\']
else
    imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN02.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN02.klb']};%0,90,180,270 degrees
    outputFolder = ['\\dm11\kellerlab\deconvolutionDatasets\15_06_11_fly_functionalImage_TM' TMstr '_Fiji\']
end

psfFilename = ['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM000001_Fiji\PSF_synthetic.tif'];

xmlFilename = ['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM' TMstr '_Fiji\dataset.xml']

transposeOrigImage = true;


%cropping purposes
ROI = [129 333  1;...
       704 1628 500];

   
%%
if( exist(outputFolder) == 0 )
    mkdir(outputFolder);
end
%%
%load PSF
PSForig = single(readTIFFstack(psfFilename));
%read Fiji transformations
tformFiji = importAffineTransformationsFiji(xmlFilename);
%load images
numImg = length(imFilenameCell);
anisotropyZ = tformFiji{1}{end}(3,3);%calibration transformation
for ii = 1:numImg
    
    %read image
    im = single(readKLBstack([imPath imFilenameCell{ii}]));        
    if( transposeOrigImage == true )
        im = permute(im, [2 1 3]);
    end

     %output image size
    if(ii == 1)
        imRefSize = size(im) .* [ 1 1 anisotropyZ];
    end
   
    %calculate weights
    imW = single(estimateDeconvolutionWeights(im, anisotropyZ , 15, []));
    
    %apply transformations
    [im,A] = applyRegistrationFromFiji(im , imRefSize, tformFiji{ii}, ROI);
    imW = applyRegistrationFromFiji(imW , imRefSize, tformFiji{ii}, ROI);        
    PSF = single(imwarp(PSForig, affine3d(A), 'interp', 'cubic'));
    
    
    %save stacks
    wFilename = [outputFolder 'simview3_TM' num2str(TM,'%6d') '_imReg_' num2str(ii) '.klb'];
    writeKLBstack(single(im), wFilename);
    wFilename = [outputFolder 'simview3_TM' num2str(TM,'%6d') '_weightsReg_' num2str(ii) '.klb'];
    writeKLBstack(single(imW), wFilename);
    wFilename = [outputFolder 'simview3_TM' num2str(TM,'%6d') '_psfReg_' num2str(ii) '.klb'];
    writeKLBstack(PSF, wFilename);
            
end
