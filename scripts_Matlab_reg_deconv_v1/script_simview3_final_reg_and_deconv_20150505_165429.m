TMvec = [ 1407 ]
experimentTag = '20150505_165429'
%%
%parameters
for TM = TMvec
    pathImPattern = ['S:\SiMView3\15-05-05\Dme_L1_57C10-GCaMP6s_' experimentTag '.corrected\SPM00\TM??????\SPM00_TM??????_' ]      
    
    TMstr = num2str(TM,'%.6d');
    baseRegistrationFolder = ['T:\temp\deconvolution\' experimentTag '_GCaMP6_TM' TMstr '\'] %where .mat files are located to read affine transformations
    
    numItersLR = 40;
    backgroundOffset = 100;
    
    PSFfilename = 'PSF_synthetic.klb';
    
    transposeOrigImage = false;
    
    %%
    %run deconvolution and registration
    function_simview3_final_registration_and_deconvolution(pathImPattern, TM, numItersLR, backgroundOffset, baseRegistrationFolder, PSFfilename, transposeOrigImage);
end