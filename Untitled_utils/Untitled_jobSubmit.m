

for TM = 2620:3172
cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /numcores:6 runMatlabJob.cmd "Y:\Exchange\Fernando\multiview-deconvolution" "script_simview3_applyAffineTransformation_15_05_12(' num2str(TM) ')"']
[rr, ss] = system(cmd);
end

%%
for TM = 2620:3172
cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /numcores:6 runMatlabJob.cmd "Y:\Exchange\Fernando\multiview-deconvolution" "script_saveResults2uint16_15_05_12(' num2str(TM) ')"']
[rr, ss] = system(cmd);
end

%%
%check they were executed correctly
ll = [];
for TM = 2500:3600
   filename = ['T:\temp\deconvolution\15_05_12_fly_functionalImage_cluster\TM00' num2str(TM) '\simview3_TM' num2str(TM) '_weightsReg_4.klb'];
   if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
%check they were executed correctly
ll = [];
for TM = 3080:3172
   filename = ['B:\keller\TM00' num2str(TM) '\simview3_TM' num2str(TM) '_test_mv_deconv_LR_multiGPU_iter40.raw'];
   if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
%check they were executed correctly
%ALL SET
ll = [];
for TM = 2620:3172
   filename = ['S:\temp\deconvolution\15_05_12_fly_functionalImage_cluster\multiGPUdeconv\simview3_TM' num2str(TM) '_test_mv_deconv_LR_multiGPU_iter40.raw'];
   if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
for TM = [1025 1030]
    cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /numcores:6 runMatlabJob.cmd "Y:\Exchange\Fernando\multiview-deconvolution" "script_saveStacksForFiji_15_06_11_JFRCcluster(' num2str(TM) ')"']
    [rr, ss] = system(cmd);    
end

%%
%check they were executed correctly
%ALL SET
ll = [];
for TM = 450:1100
   filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') 'membrCh_Fiji\dataset_weights.xml'];
   if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
%check they were executed correctly
%ALL SET
ll = [];
for TM = 450:1100
   filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') 'membrCh_Fiji\psfReg_4.klb'];
   if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
%check they were executed correctly
ll = [];
for TM = 450:1100
   %filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') 'membrCh_Fiji\ww\TP0_Channel0_Illum0_Angle3.tif'];
   filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') '_Fiji\ww\TP0_Channel0_Illum0_Angle3.tif'];
   if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
%check they were executed correctly
ll = [];
for TM = 450:1100
   %filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') 'membrCh_Fiji\weightsReg_4.klb'];
   filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') '_Fiji\weightsReg_4.klb'];
   if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
%check they were executed correctly
ll = [];
isMbr = false;
for TM = 450:1100
    if( isMbr )
        filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') 'membrCh_Fiji\weightsReg_4.klb'];
        filename2 = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') 'membrCh_Fiji\ww\TP0_Channel0_Illum0_Angle3.tif'];
        
    else
        filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') '_Fiji\weightsReg_4.klb'];
        filename2 = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') '_Fiji\ww\TP0_Channel0_Illum0_Angle3.tif'];
    end
    if( exist(filename, 'file') == 0  && exist(filename2, 'file') == 0)
       ll = [ll TM];
   end
end

%%
%check they were executed correctly
ll = [];
isMbr = true;
for TM = 450:820
    if( isMbr )
        filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') 'membrCh_Fiji\test_mv_deconv_LR_multiGPU_iter40.raw'];                
    else
        filename = ['B:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') '_Fiji\test_mv_deconv_LR_multiGPU_iter40.raw'];                
    end
    if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
for TM = 2741:3172
    cmd = ['"C:\Program Files\Microsoft HPC Pack 2008 R2\Bin\job.exe" submit /scheduler:keller-cluster.janelia.priv /user:simview /numcores:6 runMatlabJob.cmd "Y:\Exchange\Fernando\multiview-deconvolution" "script_straightenUpDrosophila_15_05_12(' num2str(TM) ')"']
    [rr, ss] = system(cmd);    
end

%%
%copy data back from cluster
for TM = 450:1100
    cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /numcores:6 runMatlabJob.cmd "Y:\Exchange\Fernando\multiview-deconvolution" "script_saveResults2uint16_15_06_11(' num2str(TM) ')"']
    [rr, ss] = system(cmd);    
end
