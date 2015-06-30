%prepare data for JFC cluster multi-GPU deconvolution
%%
%TMvec = [1 400 600 720 900, 1150, 1600]
TMvec = [1150]
isMembrChVec = [1];


for TM = TMvec
    for isMembrCh = isMembrChVec
        %%
        TMstr = num2str(TM,'%.6d');
        if( isMembrCh ~= 0 )
            imPath = ['\\dm11\kellerlab\deconvolutionDatasets\15_06_11_fly_functionalImage_TM' TMstr 'membrCh_Fiji\']
        else
            imPath = ['\\dm11\kellerlab\deconvolutionDatasets\15_06_11_fly_functionalImage_TM' TMstr '_Fiji\']
        end
        
        imFilename = {'TP0_Channel0_Illum0_Angle0', 'TP0_Channel0_Illum0_Angle1', 'TP0_Channel0_Illum0_Angle2', 'TP0_Channel0_Illum0_Angle3'};
        psfFilename = {'transfomed PSF of viewsetup 0', 'transfomed PSF of viewsetup 1', 'transfomed PSF of viewsetup 2', 'transfomed PSF of viewsetup 3'};
        
        
        %%
        %load images
        numImg = length(imFilename);
        for ii = 1:numImg
            %read images
            im = readTIFFstack([imPath imFilename{ii} '.tif']);
            PSF  = readTIFFstack([imPath psfFilename{ii} '.tif']);
            
            %resave images
            wFilename = [imPath 'simview3_TM' num2str(TM,'%6d') '_imReg_' num2str(ii) '.klb'];
            writeKLBstack(single(im), wFilename);
            wFilename = [imPath 'simview3_TM' num2str(TM,'%6d') '_psfReg_' num2str(ii) '.klb'];
            writeKLBstack(single(PSF), wFilename);
            
            %calculate contrast weights
            weights = single(estimateDeconvolutionWeights(im, 1 , 15, []));%anisotropy here is 1 since they are already interpolated
            wFilename = [imPath 'simview3_TM' num2str(TM,'%6d') '_weightsReg_' num2str(ii) '.klb'];
            writeKLBstack(weights,wFilename);
            
            
        end
    end
end
%%
