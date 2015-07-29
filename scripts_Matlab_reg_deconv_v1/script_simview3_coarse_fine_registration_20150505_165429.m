TMvec = [ 1080, 1407]
experimentTag = '20150505_165429'

%%
%parameters
for TM = TMvec
    
    
    TMstr = num2str(TM,'%.6d');
    imPath = ['S:\SiMView3\15-05-05\Dme_L1_57C10-GCaMP6s_' experimentTag '.corrected\SPM00\TM' TMstr '\'];
    imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees
    
    samplingXYZ = [0.40625, 0.40625, 6.780];%in um
    FWHMpsfZ = 6.0; %in um. full width half-max of the psf in Z
    
    outputFolder = ['T:\temp\deconvolution\' experimentTag '_GCaMP6_TM' TMstr '\']
    
    transposeOrigImage = false;
    RANSACparameterSet = 1;%1->functional imaging; 2->nuclear channel development; 3->membrane channel development
    %%
    if( exist(outputFolder) == 0 )
        mkdir(outputFolder);
    end
    
    %%
    %call registration function
    function_simview3_coarse_fine_registration(imPath, imFilenameCell, samplingXYZ, FWHMpsfZ, outputFolder, transposeOrigImage, RANSACparameterSet);
    
end