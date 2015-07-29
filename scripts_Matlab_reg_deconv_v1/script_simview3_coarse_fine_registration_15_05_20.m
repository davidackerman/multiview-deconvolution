
TMvec = [550:50:750, 2600:50:2950, 4500:50:5100, 5200:50:5400, 5850:50:7200]
%%
%parameters
for TM = TMvec
    
    
    TMstr = num2str(TM,'%.6d');
    imPath = ['S:\SiMView3\15-05-20\Dme_L1_57C10_GCaMP6s_Simul_Confocal_1Hz_20150520_201919.corrected\SPM00\TM' TMstr '\']
    imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees
    
    samplingXYZ = [0.40625, 0.40625, 6.22];%in um
    FWHMpsfZ = 6.0; %in um. full width half-max of the psf in Z
    
    outputFolder = ['T:\temp\deconvolution\15_05_20_fly_functionalImage_TM' TMstr '\']
    
    transposeOrigImage = true;
    
    %%
    if( exist(outputFolder) == 0 )
        mkdir(outputFolder);
    end
    
    %%
    %call registration function
    function_simview3_coarse_fine_registration(imPath, imFilenameCell, samplingXYZ, FWHMpsfZ, outputFolder, transposeOrigImage);
    
end