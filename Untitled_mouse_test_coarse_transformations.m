%%
%parameters

TMvec = [150]; %time points to be registered

imPathPattern = ['S:\SiMView1\15-01-06\Mmu_E1_H2BmCherryRIKEN_0_20150107_141835.corrected\']; %base folder where original images are located. ??? characters will be filled with the TM value

imFilenameCell = {['SPM00\TM??????\SPM00_TM??????_CM00_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM00_CHN00.klb'], ['SPM00\TM??????\SPM00_TM??????_CM01_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM01_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras


samplingXYZ = [0.40625, 0.40625, 2.031];%sampling in um

FWHMpsf = [0.8, 0.8, 3.0]; %theoretical full-width to half-max of the PSF in um.

outputFolderPattern = ['T:\temp\registration\mouse_15_01_06\TM??????_localMaxImgBin\'];%outputfolder for debugging purposes. Leave empty for no debugging. This folder should be visible to the cluster if you want to run it on it

transposeOrigImage = false; %true if raw data was saved in tiff, since cluster PT transposed them after saving them in KLB.


%%
%downsample images
Nviews = length(imFilenameCell);
numLevels = 2;
imCell = cell(Nviews,1);

TM = TMvec(1);
for ii = 1:Nviews
    imPath = recoverFilenameFromPattern(imPathPattern,TM);
    imFilename = recoverFilenameFromPattern(imFilenameCell{ii}, TM);
    filename = [imPath imFilename];
    imCell{ii} = readKLBstack(filename);
    imCell{ii} = stackBinning(imCell{ii},numLevels);
end

%%
%test transformations
cameraTransformCell = [10, 22, 31, 42];%CRITICAL. index indicating transformation selection based on flip and permutation to set all the views in the same x,y,z coordinate system. See function_multiview_camera_transformation for all the options.


TrCellPre = {[0 0 0], [230 13 304], [38 -8 0],[233 0 -59]};

anisotropyZ = samplingXYZ(3) / samplingXYZ(1);
imRefSize = ceil(size(imCell{1}) .* [1 1 anisotropyZ]);
figure;
for ii = 1:Nviews
    tform = function_multiview_camera_transformation(cameraTransformCell(ii), anisotropyZ, imRefSize);
    
    Tr = eye(4);
    Tr(4,1:3) = TrCellPre{ii};
    
    tform = tform * Tr;
        
    addpath './imWarpFast/'
    imAux = imwarpfast(single(imCell{ii}), tform, 0, imRefSize);
    rmpath './imWarpFast/'
    
    minI = min(imAux(:));
    maxI = max(imAux(:));
    imAux = uint16( 4096 * (imAux-minI) / (maxI-minI) );
    ff = recoverFilenameFromPattern( outputFolderPattern,TM);
    writeKLBstack(imAux,[ff 'debug_findCoarseTranformation_view' num2str(ii-1) '.klb' ]);
    
    subplot(2,2,ii); imagesc(imAux(:,:,166));title(['view ' num2str(ii-1)]);
end
