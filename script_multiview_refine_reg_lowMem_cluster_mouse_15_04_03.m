
%%
%parameters
TMvec = [0:499]; %time points to be registered

imPathPattern = ['S:\SiMView1\15-04-03\Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked\']; %base folder where original images are located. ??? characters will be filled with the TM value

disp('==============================================================================================================');
disp(['============YOU NEED TO HAVE AT LEAST ONE FILE NAMED ' imPathPattern 'MVrefine_deconv_LR_multiGPU_param_TM?????.xml CONTAINING A GOOD ALIGNMENT TO START THE PROCESS FOR THE OTHER TIME POINTS================']);
disp('==============================================================================================================');

imFilenameCell = {['SPM00\TM??????\SPM00_TM??????_CM00_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM00_CHN00.klb'], ['SPM00\TM??????\SPM00_TM??????_CM01_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM01_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras

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
isSubmitted = false(length(TMvec), 1);


while( isempty(isSubmitted == false) == false )
   
    for ii = 1:length(TMvec)
       if( isSubmitted(ii) == true )
           continue;
       end
        
    
       TODO;
       
    end
    %pause for 60 seconds to check again submitted jobs
    pause(60);
end

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
    function_multiview_refine_registration_lowMem(imPath, imFilenameCellTM, Acell, samplingXYZ, FWHMpsf, outputFolder, transposeOrigImage, RANSACparameterSet, deconvParam, TM);
    
end