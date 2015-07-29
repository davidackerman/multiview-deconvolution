%parameters

pathImPattern = 'S:\SiMView3\15-05-12\Dme_E1_57C10_GCaMP6s_20150512_134538.corrected\SPM00\TM??????\SPM00_TM??????_'

TMvec = [719 2517 1149 3169];


for TM = TMvec

TMstr = num2str(TM,'%.6d');
baseRegistrationFolder = ['T:\temp\deconvolution\15_05_12_fly_functionalImage\TM' TMstr] %where .mat files are located to read affine transformations

numItersLR = 40;
backgroundOffset = 100;

PSFfilename = 'PSF_synthetic.klb';

transposeOrigImage = false;

%%
%run deconvolution and registration
function_simview3_final_registration_and_deconvolution(pathImPattern, TM, numItersLR, backgroundOffset, baseRegistrationFolder, PSFfilename, transposeOrigImage);

end