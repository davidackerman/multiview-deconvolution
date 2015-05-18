%we assume images have been coarsely aligned with
%script_simview3_coarse_registration.m script

imPath = 'E:\simview3_deconvolution\15_04_24_fly_functionalImage\TM1445\Matlab_coarse_register_downsample2x_doubleBlurred'

imFilename = {'imRegister_Matlab_CM00', 'imRegister_Matlab_CM01', 'imRegister_Matlab_CM02', 'imRegister_Matlab_CM03'};

numHypothesis = 5;

numWorkers = 10;

minIntensityValue = 150;
blockSize = 32;
searchRadius = 64;

%%
%constant parameters

numIm = length(imFilename);

%%
%find correspondences between images pairwise
Tcell = cell(numIm, numIm);
for ii = 1:numIm
    imRef = readKLBstack([imPath '\' imFilename{ii} '.klb']);
    for jj = ii+1:numIm
        im = readKLBstack([imPath '\' imFilename{jj} '.klb']);
        Tcell{ii,jj} = pairwiseImageBlockMatching(imRef,im, blockSize, searchRadius, numHypothesis, minIntensityValue, numWorkers);
    end
end

%%
%fit affine transformation for all views
%tformCell = fitAffineMultiview(Tcell);
