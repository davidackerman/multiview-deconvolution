% we save files in disk in order to avoid high memory occupancy (even if I/O increases)
function function_multiview_refine_registration_lowMem(imPath, imFilenameCell, AcellPre, samplingXYZ, FWHMpsf, outputFolder, transposeOrigImage, RANSACparameterSet, deconvParam, TM)


%%
%fixed parameters
anisotropyZ = samplingXYZ(3) / samplingXYZ(1);

%%
%generate PSF for reference view
disp(['Generating PSF']);
PSF = generatePSF(samplingXYZ, FWHMpsf, []); 

%%
%read images and apply initial registration
Nviews = length(imFilenameCell);
tic;
imTempName = cell(Nviews,1);
for ii = 1:Nviews %parfor here is not advisable because of memory usage   
    imTempName{ii} = tempname;
    %apply coarse transformation
    disp(['Reading original image for view ' num2str(ii-1)]);
    tstart = tic;
    filename = [imPath imFilenameCell{ii}];
    im = single(readKLBstack(filename));
    disp(['Took ' num2str(toc(tstart)) ' secs']);
    
    if( ii == 1)
       imRefSize = ceil(size(im) .* [1 1 anisotropyZ]);
    end
    
    
    disp(['Applying pre-defined flip and permutation to view ' num2str(ii-1)]);
    tstart = tic;        
    %transform image    
    addpath './imWarpFast/'
    im = imwarpfast(im, AcellPre{ii}, 3, imRefSize);
    rmpath './imWarpFast/'    
    disp(['Took ' num2str(toc(tstart)) ' secs']);
    
    %saving file in temporary storage    
    disp(['Savign transformed image to preserve image for view ' num2str(ii-1)]);
    tstart = tic;
    minI = min(im(:));
    maxI = max(im(:));
    scale = 4096.0;
    imAux = uint16( scale * (im-minI) / (maxI-minI) );
    imFilenameOut = [imTempName{ii} '.klb'];
    writeKLBstack(uint16(imAux),imFilenameOut);
    clear imAux;
    save([imTempName{ii} '.mat'],'maxI', 'minI','scale');
    disp(['Took ' num2str(toc(tstart)) ' secs']);
    
    
    %save debugging information if requested
    if( isempty(outputFolder) == false )
        disp(['Saving debugging information and images to ' outputFolder]);
        %save parameters
        save([outputFolder filesep 'multiview_refine_coarse_reg.mat'],'AcellPre','imPath', 'imFilenameCell','RANSACparameterSet');
                   
        im = single(im);
        minI = min(im(:));
        maxI = max(im(:));
        imAux = uint16( 4096 * (im-minI) / (maxI-minI) );
        imFilenameOut = ['multiview_refine_reg_coarse_view' num2str(ii-1,'%.2d') '.klb'];
        writeKLBstack(uint16(imAux),[outputFolder filesep imFilenameOut]);
        clear imAux;
    end
    clear im;
end
ttCoarse = toc;

%%
%fine registration (we know we start from a good alignment)
tic;
[tformFineCell, statsRANSAC] = function_multiview_fine_registration_b_lowMem(outputFolder, imTempName, RANSACparameterSet);
ttFine = toc;

%%
%delete temporary files
for ii = 1:Nviews 
   delete([imTempName{ii} '.klb']);
   delete([imTempName{ii} '.mat']);
end

%%
disp(['Saving all the registration information to folder ' imPath]); 
%save XML file and log file be able to apply deconvolution
clear imCoarseCell;

%save workspase
save([imPath 'MVrefine_reg_workspace_TM' num2str(TM,'%.6d') '.mat']);

%save XML filename
Nviews = length(AcellPre);
Acell = cell(Nviews,1);
PSFcell = Acell;
filenameCell = Acell;
psfFilenameCell = Acell;
for ii = 1:Nviews
    Acell{ii} = AcellPre{ii} * tformFineCell{ii};
    PSFcell{ii} = single(imwarp(PSF, affine3d(Acell{ii}), 'interp', 'cubic'));
    PSFcell{ii} = PSFcell{ii} / sum(PSFcell{ii}(:));
    filenameCell{ii} = [imPath imFilenameCell{ii}];
    
    [~, name] = fileparts(filenameCell{ii});
    psfFilenameCell{ii} = [imPath name '_ref_psfReg.klb'];
    writeKLBstack(PSFcell{ii}, psfFilenameCell{ii}, -1, [], [], 0, 'Registered PSF');
end

filenameXML = [imPath 'MVrefine_deconv_LR_multiGPU_param_TM' num2str(TM,'%.6d') '.xml'];
saveRegistrationDeconvolutionParameters(filenameXML,filenameCell, psfFilenameCell, Acell, deconvParam.verbose, deconvParam);

%save log file with RANAC stats
fid = fopen([imPath 'MVrefine_reg_RANSACstats_TM' num2str(TM,'%.6d') '.txt'], 'w');
fprintf(fid,'Avg. residual = %.6f pixels\n', mean(sqrt(sum(statsRANSAC.residuals.^2,2))));
fprintf(fid,'Number of inliers = %d\n', statsRANSAC.numInliers);
fprintf(fid,'Took %.f secs for coarse alignment\n', ttCoarse);
fprintf(fid,'Took %.f secs for fine alignment\n', ttFine);

for ii = 1:Nviews
    fprintf(fid,'Coarse alignment affine matrix for view %d\n',ii);
    fprintf(fid, '%.2f\t %.2f\t %.2f\t %.2f\n', AcellPre{ii}');
    fprintf(fid,'Fine alignment affine matrix for view %d\n',ii);
    fprintf(fid, '%.2f\t %.2f\t %.2f\t %.2f\n', tformFineCell{ii}');
end

fclose(fid);