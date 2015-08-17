%%
%parameters

TMvec = [2700]; %time points to be registered

imPathPattern = ['S:\SiMView3\15-05-05\Dme_L1_57C10-GCaMP6s_20150505_165429.corrected\SPM00\TM??????\']; %base folder where original images are located. ??? characters will be filled with the TM value


imFilenameCell = {['SPM00_TM??????_CM00_CHN01.klb'], ['SPM00_TM??????_CM02_CHN00.klb'], ['SPM00_TM??????_CM01_CHN01.klb'], ['SPM00_TM??????_CM03_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras

samplingXYZ = [0.40625, 0.40625, 6.780];%sampling in um

FWHMpsf = [0.8, 0.8, 5.0]; %theoretical full-width to half-max of the PSF in um.

outputFolderPattern = ['E:\temp\Drosophila_GCamp6_20150505_TM??????\'];
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
cameraTransformCell = [10, 20, 30, 40];%CRITICAL. index indicating transformation selection based on flip and permutation to set all the views in the same x,y,z coordinate system. See function_multiview_camera_transformation for all the options.


TrCellPre = {[0 0 0], [15 -11 -31], [15 8 11],[34 -13 59]};

anisotropyZ = samplingXYZ(3) / samplingXYZ(1);
imRefSize = ceil(size(imCell{1}) .* [1 1 anisotropyZ]);
figure;
for ii = 1:Nviews
    tform = function_multiview_camera_transformation(cameraTransformCell(ii), anisotropyZ, imRefSize);
    
    Tr = eye(4);
    Tr(4,1:3) = TrCellPre{ii};
    
    tform = tform * Tr;
        
    addpath './imWarpFast/'
    imAux = imwarpfast(single(imCell{ii}), tform, 1, imRefSize);
    rmpath './imWarpFast/'
    
    minI = min(imAux(:));
    maxI = max(imAux(:));
    imAux = uint16( 4096 * (imAux-minI) / (maxI-minI) );
    ff = recoverFilenameFromPattern( outputFolderPattern,TM);
    if( exist(ff, 'dir') == 0 )
        mkdir(ff)
    end
    writeKLBstack(imAux,[ff 'debug_findCoarseTranformation_view' num2str(ii-1) '.klb' ]);
    
    subplot(2,2,ii); imagesc(imAux(:,:,166));title(['view ' num2str(ii-1)]);
end
