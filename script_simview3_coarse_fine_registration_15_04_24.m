%to align a time point in 12-04-24 simview4 dataset

%%
%parameters
TM = 7;


TMstr = num2str(TM,'%.6d');
imPath = ['S:\SiMView3\15-04-24\Dme_L1_57C10-GCaMP6s_20150424_142342.corrected\SPM00\TM' TMstr '\']
imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees

samplingXYZ = [0.40625, 0.40625, 5.2];%in um
FWHMpsfZ = 6.0; %in um. full width half-max of the psf in Z

outputFolder = ['T:\temp\deconvolution\15_04_24_fly_functionalImage\TM' TMstr '\']

%%
if( exist(outputFolder) == 0 )
    mkdir(outputFolder);
end

%%
%call registration function
function_simview3_coarse_fine_registration(imPath, imFilenameCell, samplingXYZ, FWHMpsfZ, outputFolder);