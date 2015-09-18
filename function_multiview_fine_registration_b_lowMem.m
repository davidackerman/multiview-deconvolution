function [Acell, statsRANSAC] = function_multiview_fine_registration_b_lowMem(debugFolder, imTempName, param)

%%
%constant parameters

numIm = length(imTempName);

numLevels = 2;
%%
%find correspondences between images pairwise
Tcell = cell(numIm, numIm);
for ii = 1:numIm        
    
    %load image
    disp(['Reloading view ' num2str(ii-1)]);
    tstart = tic;
    im = single(readKLBstack([imTempName{ii} '.klb']));
    pp = load([imTempName{ii} '.mat']);    
    im = single((pp.maxI-pp.minI) * im / pp.scale + pp.minI);    
    disp(['Took ' num2str(toc(tstart)) ' secs']);
    
    %bin image to speed process
    imB = stackBinning(im,numLevels);
    
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
        
        %load image
        disp(['Reloading view ' num2str(jj-1)]);
        tstart = tic;
        imJ = single(readKLBstack([imTempName{jj} '.klb']));
        pp = load([imTempName{jj} '.mat']);
        imJ = single((pp.maxI-pp.minI) * imJ / pp.scale + pp.minI);
        disp(['Took ' num2str(toc(tstart)) ' secs']);
        
        disp(['Detecting matches from view ' num2str(ii-1) ' to view ' num2str(jj-1)]);
        tstart = tic;
        Tcell{ii,jj} = pairwiseImageBlockMatching(im,imJ, param.blockSize, param.searchRadius, param.numHypothesis, interestPts(:,1:3), param.numWorkers, param.thrNCC);
        disp(['Took ' num2str(toc(tstart)) ' secs']);
    end
end

clear imJ;
clear im;

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
numViews = length(imTempName);
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
        %load image
        disp(['Reloading view ' num2str(ii-1)]);
        tstart = tic;
        im = single(readKLBstack([imTempName{ii} '.klb']));
        pp = load([imTempName{ii} '.mat']);
        im = single((pp.maxI-pp.minI) * im / pp.scale + pp.minI);
        disp(['Took ' num2str(toc(tstart)) ' secs']);
        
        if( ii == 1)
            imRefSize = size(im);
        end
        
        addpath './imWarpFast/'
        im = imwarpfast(im, Acell{ii}, 0, imRefSize);
        rmpath './imWarpFast/'
        im = single(im);
        minI = min(im(:));
        maxI = max(im(:));
        im = uint16( 4096 * (im-minI) / (maxI-minI) );        
        imFilenameOut = ['multiview_fine_reg_view' num2str(ii-1,'%.2d') '.klb'];
        writeKLBstack(uint16(im),[debugFolder filesep imFilenameOut]);
    end    
end