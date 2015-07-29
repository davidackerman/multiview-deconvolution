%%
%TMvecCell = {[ 1100],[400], [1000 2000]};
%experimentTagVec = {'20150709_170711','20150709_183937', '20150709_195932'}

TMvecCell = {[400], [1000 2000]};
experimentTagVec = {'20150709_183937', '20150709_195932'}

%%
%parameters
for ee = 1:length(experimentTagVec)
    experimentTag = experimentTagVec{ee};
    TMvec = TMvecCell{ee};
    for TM = TMvec
        
        
        TMstr = num2str(TM,'%.6d');
        imPath = ['S:\SiMView3\15-07-09\Dre_HuC_H2BGCaMP6s_0-1_' experimentTag '.corrected\SPM00\TM' TMstr '\'];
        imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees
        
        samplingXYZ = [0.40625, 0.40625, 6.070];%in um
        FWHMpsfZ = 5.0; %in um. full width half-max of the psf in Z
        
        
        outputFolder = ['T:\temp\deconvolution\' experimentTag '_GCaMP6_zebrafish\'];
        if( exist(outputFolder, 'dir') == 0 )
            mkdir(outputFolder);
        end
        
        outputFolder = [outputFolder 'TM' TMstr '\'];
        
        transposeOrigImage = false;
        RANSACparameterSet = 4;%1->functional imaging; 2->nuclear channel development; 3->membrane channel development
        %%
        if( exist(outputFolder) == 0 )
            mkdir(outputFolder);
        end
        
        %%
        %call registration function
        function_simview3_coarse_fine_registration(imPath, imFilenameCell, samplingXYZ, FWHMpsfZ, outputFolder, transposeOrigImage, RANSACparameterSet);
        
    end
end