TMvecCell = {[1800]};
experimentTagVec = {'20150710_181802'}

%%
%parameters
for ee = 1:length(experimentTagVec)
    experimentTag = experimentTagVec{ee};
    TMvec = TMvecCell{ee};
    for TM = TMvec
        pathImPattern = ['S:\SiMView3\15-07-10\Dre_HuC_H2BGCaMP6s_0-1_' experimentTag '.corrected\SPM00\TM??????\SPM00_TM??????_' ]                 
        
        TMstr = num2str(TM,'%.6d');
        baseRegistrationFolder = ['T:\temp\deconvolution\' experimentTag '_GCaMP6_zebrafish\TM' TMstr '\'] %where .mat files are located to read affine transformations        
        
        PSFfilename = 'PSF_synthetic.klb';
        
        transposeOrigImage = false;
        
        %%
        %run deconvolution and registration
        function_simview3_final_registration(pathImPattern, TM, baseRegistrationFolder, PSFfilename, transposeOrigImage);
    end
end