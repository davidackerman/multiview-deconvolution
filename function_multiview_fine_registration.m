function [Acell, statsRANSAC] = function_multiview_fine_registration(debugFolder, imCoarseCell, param)

%%
%constant parameters

numIm = length(imCoarseCell);

%%
%find correspondences between images pairwise
Tcell = cell(numIm, numIm);
for ii = 1:numIm      
    
    switch(lower(param.interestPointDetector))
        
        case 'dog'
            disp(['Generating embryo mask for view ' num2str(ii-1)]);
            tstart = tic;
            %generate mask (to get features only in the embryo): in case we also have beads
            mask = maskEmbryo(imCoarseCell{ii}, param.thrMask);
            %mask also areas with low intensity
            mask = mask & imCoarseCell{ii} > param.minIntensityValue;
            disp(['Took ' num2str(toc(tstart)) ' secs']);
            disp(['Detecting interest points with DoG for view ' num2str(ii-1)]);
            tstart = tic;
            %detect points of interest in reference image
            interestPts = detectInterestPoints_DOG(imCoarseCell{ii}, param.sigmaDOG, param.maxNumPeaks, param.thrPeakDOG, mask ,0, debugFolder, ii);
            disp(['Took ' num2str(toc(tstart)) ' secs']);
        
        case 'localmaxima'
            disp(['Detecting interest points with bining + local maxima intensity for view ' num2str(ii-1)]);
            tstart = tic;
            %bin image to speed process
            numLevelsB = 2;
            imB = stackBinning(imCoarseCell{ii},numLevelsB);
            %detect points of interest in reference image
            minDIstanceBetweenPeaks = min(40, max(size(imCoarseCell{ii})) / 50);
            interestPts = detectInterestPoints_localMaxima(imB, param.sigmaDOG, param.maxNumPeaks, param.minIntensityValue, [] ,0, debugFolder, ii, minDIstanceBetweenPeaks);
            interestPts(:,1:3) = interestPts(:,1:3) * (2^numLevelsB);            
            disp(['Took ' num2str(toc(tstart)) ' secs']);
    end
    
    %find correspondence for point of interest in the other images
    for jj = 1:numIm
        if( ii == jj )
            continue;
        end      
        disp(['Detecting matches from view ' num2str(ii-1) ' to view ' num2str(jj-1)]);
        tstart = tic;
        Tcell{ii,jj} = pairwiseImageBlockMatching(imCoarseCell{ii},imCoarseCell{jj}, param.blockSize, param.searchRadius, param.numHypothesis, interestPts(:,1:3), param.numWorkers);
        disp(['Took ' num2str(toc(tstart)) ' secs']);
    end
end

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