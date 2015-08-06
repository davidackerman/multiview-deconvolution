%%
%parameters

TMvec = [200:2:250]; %time points to be registered

imPathPattern = ['S:\SiMView1\15-04-03\Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked\']; %base folder where original images are located. ??? characters will be filled with the TM value

imFilenameCell = {['SPM00\TM??????\SPM00_TM??????_CM00_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM00_CHN00.klb'], ['SPM00\TM??????\SPM00_TM??????_CM01_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM01_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras
%imFilenameCell = {['SPM00\TM??????\SPM00_TM??????_CM00_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM00_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras

%affine transformation usually read from XML file
Acell = { [1.000000000000 0.000000000000 0.000000000000 0.000000000000 0.000000000000 1.000000000000 0.000000000000 0.000000000000 0.000000000000 0.000000000000 4.999384615385 0.000000000000 0.000000000000 0.000000000000 0.000000000000 1.000000000000],...
      [-0.002824110184 -0.017232392687 -4.997580466548 1626.176313855931 0.003642980848 0.999677155382 -0.144708885966 21.420803404247 1.019529021672 -0.004955497720 -0.032905931258 -359.338649285432 0.000000000000 0.000000000000 0.000000000000 1.000000000000],...
      [-1.001530991982 -0.009273726643 0.102933365955 2094.223840836948 -0.003040916543 0.995414049699 0.051263200570 27.981896339104 -0.000749875562 -0.000317531002 4.962486160228 4.743383866147 0.000000000000 0.000000000000 0.000000000000 1.000000000000],...
      [-0.000780332931 -0.015716324107 -4.971113677178 1618.834811524733 -0.003405985152 0.998164471680 -0.138298284855 57.063670726678 -1.008081872156 -0.000525990445 -0.061694656855 1772.707456825679 0.000000000000 0.000000000000 0.000000000000 1.000000000000]};
% % Acell = { [1.000000000000 0.000000000000 0.000000000000 0.000000000000 0.000000000000 1.000000000000 0.000000000000 0.000000000000 0.000000000000 0.000000000000 4.999384615385 0.000000000000 0.000000000000 0.000000000000 0.000000000000 1.000000000000],...
% %       [-0.002824110184 -0.017232392687 -4.997580466548 1626.176313855931 0.003642980848 0.999677155382 -0.144708885966 21.420803404247 1.019529021672 -0.004955497720 -0.032905931258 -359.338649285432 0.000000000000 0.000000000000 0.000000000000 1.000000000000] };



samplingXYZ = [0.40625, 0.40625, 2.031];%sampling in um

FWHMpsf = [0.8, 0.8, 3.0]; %theoretical full-width to half-max of the PSF in um.

outputFolderPattern = ['T:\temp\registration\mouse_15_04_03\refReg_TM??????\'];%outputfolder for debugging purposes. Leave empty for no debugging. This folder should be visible to the cluster if you want to run it on it

transposeOrigImage = false; %true if raw data was saved in tiff, since cluster PT transposed them after saving them in KLB.


%%
%RANSAC
%critical parameters for RANSAC alignment> This is a refinement, so parameters can be tight
RANSACparameterSet.minIntensityValue = 150; %global threshold. Any pixel below that intesnity will not be considered a point of interest for matching
RANSACparameterSet.blockSize = 96;         %blocks size (in pixels) around points of interest to match between views. The larger it is the more memory is required but the easier it is to match
RANSACparameterSet.searchRadius = 16;      %maximum distance (in pixels) between two views to match corresponding points after coarse alignment. If coarse alignment works well, this can be small. The smaller the value, the less memory is required.

%usually "stable" parameters for RANSAC alignment
RANSACparameterSet.numHypothesis = 3;       %number of possible matches for each point of interest
RANSACparameterSet.thrNCC = 0.5;            %threshold of NCC to accept a match
RANSACparameterSet.numWorkers = -1;         %set to -1 to use as many as possible. If code runs out of memory, reduce the number.
RANSACparameterSet.maxNumPeaks = 100;       %maximum number of points of interest per view to match. The higher the number the longer the code takes
RANSACparameterSet.sigmaDOG = 3;          %size of the neighborhood to detect local maxima

RANSACparameterSet.minDIstanceBetweenPeaks = 40; %minimum distance between local maxima (in pixels) to make sure we have points in all the specimen

RANSACparameterSet.numTrialsRANSAC = 50000; %number of RANSAC trials
RANSACparameterSet.maxRadiusResidualInPixels = 15.0;    %maximum residual (in pixels) to consider a RANSAC inlier



%%
%deconvolution parameters: they are not needed for registration but they are needed to
%generate the XML file that will be used by the multi-GPU code. You can
%always edit manually the XML file to change them
deconvParam.verbose = 0; %set >0 to print out intermedate deconvolution steps for debugging
deconvParam.lambdaTV = 0.0001; %Ttotal variation regularization lambda parameter. Set to < 0 to deactivate total variation regularization
deconvParam.numIter = 40; %number of Lucy-Richardson iterations
deconvParam.imBackground = 100.0; %image background level. It will be subtracted from images to make sure noise statistics are as Poissonian as possible


%%
%parameters
for TM = TMvec
        
    %%
    outputFolder = recoverFilenameFromPattern(outputFolderPattern, TM);
    if( exist(outputFolder, 'dir') == 0 )
        mkdir(outputFolder);
    end
    
    imPath = recoverFilenameFromPattern(imPathPattern, TM);
    
    imFilenameCellTM = cell(length(imFilenameCell),1);
    for ii = 1:length(imFilenameCell)
        imFilenameCellTM{ii} = recoverFilenameFromPattern(imFilenameCell{ii},TM);
        Acell{ii} = reshape(Acell{ii},[4 4]);
    end
    
    
    %select number of workers
    if( RANSACparameterSet.numWorkers < 1 )
        qq = feature('numCores');
        RANSACparameterSet.numWorkers = qq;
    end
    RANSACparameterSet.numWorkers = min(12, RANSACparameterSet.numWorkers);%12 is the maximum allowed in current version
    
    %%
    %TODO: add a flag to run this in the cluster usign job submit (like in clusterPT)
    %call registration function
    function_multiview_refine_registration(imPath, imFilenameCellTM, Acell, samplingXYZ, FWHMpsf, outputFolder, transposeOrigImage, RANSACparameterSet, deconvParam, TM);
    
end