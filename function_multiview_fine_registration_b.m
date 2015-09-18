function [Acell, statsRANSAC] = function_multiview_fine_registration_b(debugFolder, imCoarseCell, param)

%%
%constant parameters

numIm = length(imCoarseCell);

numLevels = 2;
%%
%find correspondences between images pairwise
Tcell = cell(numIm, numIm);
for ii = 1:numIm        
    
    %bin image to speed process
    imB = stackBinning(imCoarseCell{ii},numLevels);
    
    disp(['Detecting interest points using binned local maxima for view ' num2str(ii-1)]);
    tstart = tic;
    %detect points of interest in reference image
    interestPts = detectInterestPoints_localMaxima(imB, param.sigmaDOG, param.maxNumPeaks, param.minIntensityValue, [] ,0, debugFolder, ii, param.minDIstanceBetweenPeaks);    
    interestPts(:,1:3) = interestPts(:,1:3) * (2^numLevels);    
    disp(['Took ' num2str(toc(tstart)) ' secs']);
    
    %find correspondence for point of interest in the other images
    for jj = 1:numIm
        if( ii == jj )
            continue;
        end      
        disp(['Detecting matches from view ' num2str(ii-1) ' to view ' num2str(jj-1)]);
        tstart = tic;
        Tcell{ii,jj} = pairwiseImageBlockMatching(imCoarseCell{ii},imCoarseCell{jj}, param.blockSize, param.searchRadius, param.numHypothesis, interestPts(:,1:3), param.numWorkers, param.thrNCC);
        disp(['Took ' num2str(toc(tstart)) ' secs']);
    end
end


%put it in the base workspace so it is saved
assignin('caller','Tcell',Tcell);

%%
%fit affine transformation for all views
disp(['Running multiview RANSAC alignment']);
tstart = tic;
[AcellRansac, statsCell] = fitAffineMultiviewRANSAC(Tcell, param.maxRadiusResidualInPixels, param.numTrialsRANSAC, param.numWorkers);
disp(['Took ' num2str(toc(tstart)) ' secs']);

%%
%select best RANSAC match
[idxMaxInliers, idxMinAvgResidual] = parseRANSACstats(statsCell);
%%disp(['Avg. residual = ' num2str(mean(sqrt(sum(statsCell{idxMaxInliers}.residuals.^2,2)))) ' pixels for ' num2str(statsCell{idxMaxInliers}.numInliers) ' inliers'])
statsRANSAC = statsCell{idxMaxInliers};

%collect final transform matrices
A = AcellRansac{idxMaxInliers};%final affines transformation
numViews = length(imCoarseCell);
Acell = cell(numViews,1);
Acell{1} = eye(4);
for ii = 2:numViews
   Acell{ii} = [reshape(A(12 * (ii-2) + 1: 12 *(ii-1)),[4 3]), [0;0;0;1]]; 
end

%%
%save debugging information if requested
if( isempty(debugFolder) == false )
    disp(['Saving debugging information and images to ' debugFolder]);
    %save parameters    
    save([debugFolder filesep 'multiview_fine_reg.mat'],'Acell','Tcell', 'statsRANSAC');        
    %apply transformation to each stack
    %parfor here can run out of memory for full resolution plus imwarp is already multi-thread (about 50% core usage)
    for ii = 1:numViews
        addpath './imWarpFast/'
        im = imwarpfast(imCoarseCell{ii}, Acell{ii}, 0, size(imCoarseCell{1}));
        rmpath './imWarpFast/'
        im = single(im);
        minI = min(im(:));
        maxI = max(im(:));
        im = uint16( 4096 * (im-minI) / (maxI-minI) );        
        imFilenameOut = ['multiview_fine_reg_view' num2str(ii-1,'%.2d') '.klb'];
        writeKLBstack(uint16(im),[debugFolder filesep imFilenameOut]);
    end    
end