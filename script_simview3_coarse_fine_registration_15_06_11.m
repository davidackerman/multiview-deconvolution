
TMvec = [600, 720, 900, 1150, 1600]
%%
%parameters
for TM = TMvec
    
    
    TMstr = num2str(TM,'%.6d');
    imPath = ['S:\SiMView3\15-06-11\Dme_E2_His2AvRFP_spiderGFP_12-03_20150611_155054.corrected\SPM00\TM' TMstr '\']
    imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees
    
    samplingXYZ = [0.40625, 0.40625, 3.250];%in um
    FWHMpsfZ = 4.0; %in um. full width half-max of the psf in Z
    
    outputFolder = ['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM' TMstr '\']
    
    transposeOrigImage = true;
    
    %%
    if( exist(outputFolder) == 0 )
        mkdir(outputFolder);
    end
    
    %%
    %call registration function
    function_simview3_coarse_fine_registration(imPath, imFilenameCell, samplingXYZ, FWHMpsfZ, outputFolder, transposeOrigImage);
    
end