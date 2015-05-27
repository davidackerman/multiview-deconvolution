%parameters

pathImPattern = 'S:/SiMView3/15-04-24/Dme_L1_57C10-GCaMP6s_20150424_142342.corrected/SPM00/TM??????/SPM00_TM??????_'
TM = 7

baseRegistrationFolder = 'T:\temp\deconvolution\15_04_24_fly_functionalImage\TM000007'

numItersLR = 40;
backgroundOffset = 102;

PSFfilename = 'PSF_synthetic.klb';

transposeOrigImage = false;

%%
%run deconvolution and registration
function_simview3_final_registration_and_deconvolution(pathImPattern, TM, numItersLR, backgroundOffset, baseRegistrationFolder, PSFfilename, transposeOrigImage);