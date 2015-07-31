function function_multiview_coarse_fine_registration(imPath, imFilenameCell, cameraTransformCell, samplingXYZ, FWHMpsf, outputFolder, transposeOrigImage, RANSACparameterSet, deconvParam)


%%
%fixed parameters
anisotropyZ = samplingXYZ(3) / samplingXYZ(1);

%%
%generate PSF for reference view
disp(['Generating PSF']);
PSF = generatePSF(samplingXYZ, FWHMpsf, []); 


%%
%coarse registration (basically flipdim + permute + scale)
tic;
[imCoarseCell, tformCoarseCell] = function_multiview_coarse_registration(imPath, imFilenameCell, cameraTransformCell, PSF, anisotropyZ, outputFolder, transposeOrigImage);
ttCoarse = toc;

%%
%fine registration
tic;
[tformFineCell, statsRANSAC] = function_multiview_fine_registration(outputFolder, imCoarseCell, RANSACparameterSet);
ttFine = toc;

%%
disp(['Saving all the registration information to folder ' imPath]); 
%save XML file and log file be able to apply deconvolution
clear imCoarseCell;

%save workspase
save([imPath 'MVreg_workspace.mat']);

%save XML filename
Nviews = length(tformCoarseCell);
Acell = cell(Nviews,1);
PSFcell = Acell;
filenameCell = Acell;
psfFilenameCell = Acell;
for ii = 1:Nviews
    Acell{ii} = tformCoarseCell{ii} * tformFineCell{ii};
    PSFcell{ii} = single(imwarp(PSF, affine3d(Acell{ii}), 'interp', 'cubic'));
    PSFcell{ii} = PSFcell{ii} / sum(PSFcell{ii}(:));
    filenameCell{ii} = [imPath imFilenameCell{ii}];
    
    [~, name] = fileparts(filenameCell{ii});
    psfFilenameCell{ii} = [imPath name '_psfReg.klb'];
    writeKLBstack(PSFcell{ii}, psfFilenameCell{ii}, -1, [], [], 0, 'Registered PSF');
end

filenameXML = [imPath 'MVdeconv_LR_multiGPU_param.xml'];
saveRegistrationDeconvolutionParameters(filenameXML,filenameCell, psfFilenameCell, Acell, deconvParam.verbose, deconvParam);

%save log file with RANAC stats
fid = fopen([imPath 'MVreg_RANSACstats.txt'], 'w');
fprintf(fid,'Avg. residual = %.6f pixels', num2str(mean(sqrt(sum(statsRANSAC.residuals.^2,2)))));
fprintf(fid,'Number of inliers = %d', num2str(statsRANSAC.numInliers));
fprintf(fid,'Took %.f secs for coarse alignment\n', ttCoarse);
fprintf(fid,'Took %.f secs for fine alignment\n', ttFine);

for ii = 1:Nviews
    fprintf(fid,'Coarse alignment affine matrix for view %d\n',ii);
    fprintf(fid, '%.2f\t %.2f\t %.2f\t %.2f\n', tformCoarseCell{ii}');
    fprintf(fid,'Fine alignment affine matrix for view %d\n',ii);
    fprintf(fid, '%.2f\t %.2f\t %.2f\t %.2f\n', tformFineCell{ii}');
end

fclose(fid);