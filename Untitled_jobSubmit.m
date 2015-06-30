

for TM = 2620:3172
cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /numcores:6 runMatlabJob.cmd "Y:\Exchange\Fernando\multiview-deconvolution" "script_simview3_applyAffineTransformation_15_05_12(' num2str(TM) ')"']
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
for TM = 2633:2762
   filename = ['B:\keller\TM00' num2str(TM) '\simview3_TM' num2str(TM) '_test_mv_deconv_LR_multiGPU_iter40.raw'];
   if( exist(filename, 'file') == 0 )
       ll = [ll TM];
   end
end

%%
for TM = 450:1100
    cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /numcores:6 runMatlabJob.cmd "Y:\Exchange\Fernando\multiview-deconvolution" "script_saveStacksForFiji_15_06_11_JFRCcluster(' num2str(TM) ',0)"']
    [rr, ss] = system(cmd);
    cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /numcores:6 runMatlabJob.cmd "Y:\Exchange\Fernando\multiview-deconvolution" "script_saveStacksForFiji_15_06_11_JFRCcluster(' num2str(TM) ',1)"']
    [rr, ss] = system(cmd);
end

