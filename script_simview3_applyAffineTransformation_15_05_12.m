%to be called from cluster
%TMvec = [2500:3600];
function script_simview3_applyAffineTransformation_15_05_12(TM)

%parameters
pathImPattern = 'S:\SiMView3\15-05-12\Dme_E1_57C10_GCaMP6s_20150512_134538.corrected\SPM00\TM??????\SPM00_TM??????_'


TMstr = num2str(TM,'%.6d');
baseRegistrationFolder = ['T:\temp\deconvolution\15_05_12_fly_functionalImage_cluster\TM' TMstr] %where .mat files are located to read affine transformations


PSFfilename = 'T:\temp\deconvolution\15_05_12_fly_functionalImage_cluster\PSF_synthetic.klb';

transposeOrigImage = false;

anisotropyZ = 14.1785;

%%
%run deconvolution and registration
function_simview3_apply_final_registration(pathImPattern, TM, baseRegistrationFolder, PSFfilename, transposeOrigImage, anisotropyZ);

