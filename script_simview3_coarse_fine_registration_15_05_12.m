
%%
%parameters
TM = 39;


TMstr = num2str(TM,'%.6d');
imPath = ['S:\SiMView3\15-05-12\Dme_E1_57C10_GCaMP6s_20150512_134538.corrected\SPM00\TM' TMstr '\']
imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees

samplingXYZ = [0.40625, 0.40625, 5.76];%in um
FWHMpsfZ = 6.0; %in um. full width half-max of the psf in Z

outputFolder = ['T:\temp\deconvolution\15_05_12_fly_functionalImage\TM' TMstr '\']

transposeOrigImage = false;

%%
if( exist(outputFolder) == 0 )
    mkdir(outputFolder);
end

%%
%call registration function
function_simview3_coarse_fine_registration(imPath, imFilenameCell, samplingXYZ, FWHMpsfZ, outputFolder, transposeOrigImage);