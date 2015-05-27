function simview3_fine_registration(imPath, imFilename)

numHypothesis = 3;
numWorkers = min(12, feature('numCores'));

minIntensityValue = 150;
blockSize = 144;%96;%critical to make sure NCC discriminates enough
searchRadius = 64 * 2;

%it is better to find 
maxNumPeaks = 100;
sigmaDOG = 3.0 * 2;
thrPeakDOG = 15;


numTrialsRANSAC = 50000;
maxRadiusResidualInPixels = 15.0;

%%
%constant parameters

numIm = length(imFilename);

%%
%find correspondences between images pairwise
Tcell = cell(numIm, numIm);
for ii = 1:numIm
    imRef = readKLBstack([imPath '\' imFilename{ii} '.klb']);
    
    %detect points of interest in reference image
    interestPts = detectInterestPoints_DOG(imRef, sigmaDOG, maxNumPeaks, thrPeakDOG, imRef > minIntensityValue,0);
    
    %find correspondence for point of interest in the other images
    for jj = 1:numIm
        if( ii == jj )
            continue;
        end
        im = readKLBstack([imPath '\' imFilename{jj} '.klb']);
        Tcell{ii,jj} = pairwiseImageBlockMatching(imRef,im, blockSize, searchRadius, numHypothesis, interestPts(:,1:3), numWorkers);
    end
end

%%
%fit affine transformation for all views
[AcellRansac, statsCell] = fitAffineMultiviewRANSAC(Tcell, maxRadiusResidualInPixels, numTrialsRANSAC, numWorkers);

%%
%select best RANSAC match
[idxMaxInliers, idxMinAvgResidual] = parseRANSACstats(statsCell);
disp(['Avg. residual = ' num2str(mean(sqrt(sum(statsCell{idxMaxInliers}.residuals.^2,2)))) ' pixels for ' num2str(statsCell{idxMaxInliers}.numInliers) ' inliers'])

%collect final transform matrices
A = AcellRansac{idxMaxInliers};%final affines transformation
numViews = length(imFilename);
Acell = cell(numViews,1);
Acell{1} = eye(4);
for ii = 2:numViews
   Acell{ii} = [reshape(A(12 * (ii-2) + 1: 12 *(ii-1)),[4 3]), [0;0;0;1]]; 
   Acell{ii}
end

%save transform
tformCell = Acell;
save([imPath '\imWarp_Matlab_tform_fine.mat'],'tformCell', 'Tcell', 'imPath', 'imFilename');
%%
%apply transformation to each stack
%parfor here can run out of memory for full resolution plus imwarp is already multi-thread (about 50% core usage)
for ii = 1:numViews       
    im = readKLBstack([imPath '\' imFilename{ii} '.klb']);
    im = imwarp(im, affine3d(Acell{ii}), 'Outputview', imref3d(size(im)), 'interp', 'linear');
    
    writeKLBstack(uint16(im),[imPath '\' imFilename{ii} '_regRANSAC.klb']);
end
