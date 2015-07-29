eSize = 1358954496;
missingTMnuc = [];
missingTMmem = [];
for TM = 450:1100
    filenameIn = ['\\fxt.int.janelia.org\nobackup\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') '_Fiji\test_mv_deconv_LR_multiGPU_iter40.raw'];
    filenameOut = ['S:\temp\deconvolution\15_06_11_dualChannel\multiGPUdeconv_nuc\simview3_TM' num2str(TM,'%.4d') '_test_mv_deconv_LR_multiGPU_iter40.raw'];
    
    if( exist(filenameOut,'file') )
        D = dir(filenameOut);
        if( D(1).bytes == eSize )
            continue;%we already have the file
        end
    end
    
    if(exist(filenameIn,'file') )
        D = dir(filenameIn);
        if( D(1).bytes == eSize )
            %copyfile
            copyfile(filenameIn, filenameOut);
            copyfile([filenameIn '.txt'], [filenameOut '.txt']);
            continue;
        end
    end
    
    %file is missing
    missingTMnuc = [missingTMnuc TM];
end


for TM = 450:1100
    filenameIn = ['\\fxt.int.janelia.org\nobackup\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') 'membrCh_Fiji\test_mv_deconv_LR_multiGPU_iter40.raw'];
    filenameOut = ['S:\temp\deconvolution\15_06_11_dualChannel\multiGPUdeconv_mem\simview3_TM' num2str(TM,'%.4d') '_test_mv_deconv_LR_multiGPU_iter40.raw'];
    
    if( exist(filenameOut,'file') )
        D = dir(filenameOut);
        if( D(1).bytes == eSize )
            continue;%we already have the file
        end
    end
    
    if(exist(filenameIn,'file') )
        D = dir(filenameIn);
        if( D(1).bytes == eSize )
            %copyfile
            copyfile(filenameIn, filenameOut);
            copyfile([filenameIn '.txt'], [filenameOut '.txt']);
            continue;
        end
    end
    
    %file is missing
    missingTMmem = [missingTMmem TM];
end


missingTMnuc

missingTMmem