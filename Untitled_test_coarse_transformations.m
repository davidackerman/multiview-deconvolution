%%
%parameters

TMvec = [1]; %time points to be registered

imPathPattern = ['S:\SiMView3\15-08-24\Dme_E1_His2AvRFP_01234567_diSPIM_20150824_220200.corrected\SPM00\TM??????\']; %base folder where original images are located. ??? characters will be filled with the TM value

imFilenameCell = {['SPM00_TM??????_CM00_CHN01.klb'],['SPM00_TM??????_CM02_CHN00.klb'],['SPM00_TM??????_CM02_CHN04.klb'],['SPM00_TM??????_CM01_CHN05.klb'],['SPM00_TM??????_CM01_CHN07.klb'],['SPM00_TM??????_CM03_CHN06.klb'],['SPM00_TM??????_CM03_CHN02.klb'],['SPM00_TM??????_CM00_CHN03.klb']};



samplingXYZ = [0.40625, 0.40625, 1.625];%sampling in um
FWHMpsf = [0.8, 0.8, 4.0]; %theoretical full-width to half-max of the PSF in um.

outputFolderPattern = ['E:\temp\diSPIM_20150824_220200_TM??????\'];
transposeOrigImage = false; %true if raw data was saved in tiff, since cluster PT transposed them after saving them in KLB.


%%
%downsample images
Nviews = length(imFilenameCell);
numLevels = 1;
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
cameraTransformCell = [10, 21, 41, 30,30, 41, 41, 10];%CRITICAL. index indicating transformation selection based on flip and permutation to set all the views in the same x,y,z coordinate system. See function_multiview_camera_transformation for all the options.

TrCellPre = {[0 0 0], [0 0 0], [0 0 0], [0 0 0],[0 0 0], [0 0 0], [0 0 0], [0 0 0]};  %Translation for each camera after the camera transform cell in order to perform coarse alignmnet. Leave empty if you want to automatically calculate it using normalized cross correlation

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
    
    subplot(4,2,ii); imagesc(imAux(:,:,166));title(['view ' num2str(ii-1)]);
end
