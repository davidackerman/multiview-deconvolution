%parameters


pathImPattern = ['S:/SiMView3/15-06-11/Dme_E2_His2AvRFP_spiderGFP_12-03_20150611_155054.corrected/SPM00/TM??????/SPM00_TM??????_']

TM = 1

TMstr = num2str(TM,'%.6d');
baseRegistrationFolder = ['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM' TMstr]

numItersLR = 40;
backgroundOffset = 100;

PSFfilename = 'PSF_synthetic.klb';

transposeOrigImage = true;

%%
%run deconvolution and registration
function_simview3_final_registration_and_deconvolution(pathImPattern, TM, numItersLR, backgroundOffset, baseRegistrationFolder, PSFfilename, transposeOrigImage);