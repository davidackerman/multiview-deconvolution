%parameters

pathImPattern = 'S:\SiMView3\15-05-22\Dme_E1_57C10_GCaMP6s_Sequential_20150522_155707.corrected\SPM00\TM??????\SPM00_TM??????_'

TM = 3

TMstr = num2str(TM,'%.6d');
baseRegistrationFolder = ['T:\temp\deconvolution\20150522_155707_fly_with_beads_TM' TMstr] %where .mat files are located to read affine transformations

numItersLR = 40;
backgroundOffset = 100;

PSFfilename = 'PSF_synthetic.klb';

transposeOrigImage = true;

%%
%run deconvolution and registration
function_simview3_final_registration_and_deconvolution(pathImPattern, TM, numItersLR, backgroundOffset, baseRegistrationFolder, PSFfilename, transposeOrigImage);