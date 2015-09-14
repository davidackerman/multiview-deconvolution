%%
%parameters

TMvec = [190]; %time points to be registered

imPathPattern = ['S:\SiMView1\15-08-10\Mmu_E1_mKate2_20150810_160708.corrected\']; %base folder where original images are located. ??? characters will be filled with the TM value


imFilenameCell = {['SPM00\TM??????\SPM00_TM??????_CM00_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM00_CHN00.klb'], ['SPM00\TM??????\SPM00_TM??????_CM01_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM01_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras


%affine transformation usually read from XML file
Acell = { [1.00000000,0.00000000,0.00000000,0.00000000,0.00000000,1.00000000,0.00000000,0.00000000,0.00000000,0.00000000,4.99938462,0.00000000,0.00000000,0.00000000,0.00000000,1.00000000],...
      [-0.02407608,-0.00948852,4.13857822,220.47317953,0.01461225,0.99115242,0.01387963,-13.03071285,-1.25404153,0.01388313,0.17170518,1950.27592110,0.00000000,0.00000000,0.00000000,1.00000000],...
      [-0.97796366,-0.00081405,0.22740175,1831.77891044,0.00193885,0.99559783,-0.04890357,20.76365333,-0.00491596,0.00087792,5.12509772,-19.48375644,0.00000000,0.00000000,0.00000000,1.00000000],...
      [0.01414674,-0.01729993,4.03772341,194.10249454,-0.02028785,0.99865400,0.05649515,15.44759169,1.22871975,0.00894557,-0.01399158,-461.85669938,0.00000000,0.00000000,0.00000000,1.00000000]};



samplingXYZ = [0.325, 0.325, 2.031];%sampling in um

FWHMpsf = [0.65, 0.65, 5.0]; %theoretical full-width to half-max of the PSF in um.

outputFolderPattern = ['T:\temp\registration\mouse_15_08_10\refReg_TM??????\'];%outputfolder for debugging purposes. Leave empty for no debugging. This folder should be visible to the cluster if you want to run it on it

transposeOrigImage = false; %true if raw data was saved in tiff, since cluster PT transposed them after saving them in KLB.


%%
%RANSAC
%critical parameters for RANSAC alignment> This is a refinement, so parameters can be tight
RANSACparameterSet.minIntensityValue = 150; %global threshold. Any pixel below that intesnity will not be considered a point of interest for matching
RANSACparameterSet.blockSize = 96;         %blocks size (in pixels) around points of interest to match between views. The larger it is the more memory is required but the easier it is to match
RANSACparameterSet.searchRadius = 32;      %maximum distance (in pixels) between two views to match corresponding points after coarse alignment. If coarse alignment works well, this can be small. The smaller the value, the less memory is required.

%usually "stable" parameters for RANSAC alignment
RANSACparameterSet.numHypothesis = 3;       %number of possible matches for each point of interest
RANSACparameterSet.thrNCC = 0.6;            %threshold of NCC to accept a match
RANSACparameterSet.numWorkers = -1;         %set to -1 to use as many as possible. If code runs out of memory, reduce the number.
RANSACparameterSet.maxNumPeaks = 150;       %maximum number of points of interest per view to match. The higher the number the longer the code takes
RANSACparameterSet.sigmaDOG = 3;          %size of the neighborhood to detect local maxima

RANSACparameterSet.minDIstanceBetweenPeaks = 40; %minimum distance between local maxima (in pixels) to make sure we have points in all the specimen

RANSACparameterSet.numTrialsRANSAC = 50000; %number of RANSAC trials
RANSACparameterSet.maxRadiusResidualInPixels = 15.0;    %maximum residual (in pixels) to consider a RANSAC inlier



%%
%deconvolution parameters: they are not needed for registration but they are needed to
%generate the XML file that will be used by the multi-GPU code. You can
%always edit manually the XML file to change them
deconvParam.verbose = 0; %set >0 to print out intermedate deconvolution steps for debugging
deconvParam.lambdaTV = 0.0000; %Ttotal variation regularization lambda parameter. Set to < 0 to deactivate total variation regularization
deconvParam.numIter = 40; %number of Lucy-Richardson iterations
deconvParam.imBackground = 100.0; %image background level. It will be subtracted from images to make sure noise statistics are as Poissonian as possible
deconvParam.blockZsize = 512; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.

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