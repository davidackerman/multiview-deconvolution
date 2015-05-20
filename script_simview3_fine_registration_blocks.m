%we assume images have been coarsely aligned with
%script_simview3_coarse_registration.m script

imPath = 'E:\simview3_deconvolution\15_04_24_fly_functionalImage\TM1445\Matlab_coarse_register_downsample2x_doubleBlurred'

imFilename = {'imRegister_Matlab_CM00', 'imRegister_Matlab_CM01', 'imRegister_Matlab_CM02', 'imRegister_Matlab_CM03'};

numHypothesis = 5;
numWorkers = 10;

minIntensityValue = 150;
blockSize = 96;%critical to make sure NCC discriminates enough
searchRadius = 64;

maxNumPeaks = 100;
sigmaDOG = 3.0;
thrPeakDOG = 15;


numTrialsRANSAC = 50000;
maxRadiusResidualInPixels = 10.0;

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
[Acell, statsCell] = fitAffineMultiviewRANSAC(Tcell, maxRadiusResidualInPixels, numTrialsRANSAC, numWorkers);
