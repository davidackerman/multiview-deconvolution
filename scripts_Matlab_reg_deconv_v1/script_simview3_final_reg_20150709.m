%TMvecCell = {[ 1100],[400], [1000 2000]};
%experimentTagVec = {'20150709_170711','20150709_183937', '20150709_195932'}

TMvecCell = {[1000 2000]};
experimentTagVec = {'20150709_195932'}

%ROI = [81 1 1;944 2048 972];%to make them power of 2 and 3

ROI = [24 1 1;752 2048 1152];
%%
%parameters
for ee = 1:length(experimentTagVec)
    experimentTag = experimentTagVec{ee};
    TMvec = TMvecCell{ee};
    for TM = TMvec
        pathImPattern = ['S:\SiMView3\15-07-09\Dre_HuC_H2BGCaMP6s_0-1_' experimentTag '.corrected\SPM00\TM??????\SPM00_TM??????_' ]                 
        
        TMstr = num2str(TM,'%.6d');
        baseRegistrationFolder = ['T:\temp\deconvolution\' experimentTag '_GCaMP6_zebrafish\TM' TMstr '\'] %where .mat files are located to read affine transformations        
        
        PSFfilename = 'PSF_synthetic.klb';
        
        transposeOrigImage = false;
        
        %%
        %run deconvolution and registration
        function_simview3_final_registration(pathImPattern, TM, baseRegistrationFolder, PSFfilename, transposeOrigImage, ROI);
    end
end