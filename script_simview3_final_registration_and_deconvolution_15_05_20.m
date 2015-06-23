%parameters

pathImPattern = 'S:\SiMView3\15-05-20\Dme_L1_57C10_GCaMP6s_Simul_Confocal_1Hz_20150520_201919.corrected\SPM00\TM??????\SPM00_TM??????_'

TM = 4548

TMstr = num2str(TM,'%.6d');
baseRegistrationFolder = ['T:\temp\deconvolution\15_05_20_fly_functionalImage_TM' TMstr] %where .mat files are located to read affine transformations

numItersLR = 40;
backgroundOffset = 100;

PSFfilename = 'PSF_synthetic.klb';

transposeOrigImage = true;

%%
%run deconvolution and registration
function_simview3_final_registration_and_deconvolution(pathImPattern, TM, numItersLR, backgroundOffset, baseRegistrationFolder, PSFfilename, transposeOrigImage);