%prepare data for JFC cluster multi-GPU deconvolution
%%
TMvec = [1 400 600 720 900, 1150, 1600]
isMembrChVec = [0 1];

itersVec = [20 40 70 100];

for TM = TMvec
    for isMembrCh = isMembrChVec
        for iters = itersVec
            %%
            TMstr = num2str(TM,'%.6d');
            if( isMembrCh ~= 0 )
                imPath = ['//dm11/kellerlab/deconvolutionDatasets/15_06_11_fly_functionalImage_TM' TMstr 'membrCh_Fiji/simview3_TM' num2str(TM) '_']
            else
                imPath = ['//dm11/kellerlab/deconvolutionDatasets/15_06_11_fly_functionalImage_TM' TMstr '_Fiji/simview3_TM' num2str(TM) '_']
            end
            
            cmd = ['C:\Users\Fernando\matlabProjects\deconvolution\CUDA\build\test\Release\test_multiview_deconvolution_LR_multiGPU.exe ' imPath ' ' num2str(iters)]
            
            tic;
            [rr,ss] = system(cmd);
            ss
            toc
            
            
        end
    end
end


