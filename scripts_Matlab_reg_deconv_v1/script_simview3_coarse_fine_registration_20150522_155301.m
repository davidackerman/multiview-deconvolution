
%%
%parameters
TM = 113;


TMstr = num2str(TM,'%.6d');
imPath = ['S:\SiMView3\15-05-22\Dme_E1_57C10_GCaMP6s_Simultaneous_20150522_155301.corrected\SPM00\TM' TMstr '\']
imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees

samplingXYZ = [0.40625, 0.40625, 6.22];%in um
FWHMpsfZ = 6.0; %in um. full width half-max of the psf in Z

outputFolder = ['T:\temp\deconvolution\20150522_155301_fly_with_beads_TM' TMstr '\']

transposeOrigImage = true;

%%
if( exist(outputFolder) == 0 )
    mkdir(outputFolder);
end

%%
%call registration function
function_simview3_coarse_fine_registration(imPath, imFilenameCell, samplingXYZ, FWHMpsfZ, outputFolder, transposeOrigImage);