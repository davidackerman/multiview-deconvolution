%%
%parameters

TMvec = [1]; %time points to be registered

imPathPattern = ['S:\SiMView3\15-08-24\Dme_E1_His2AvRFP_01234567_diSPIM_20150824_220200.corrected\SPM00\TM??????\']; %base folder where original images are located. ??? characters will be filled with the TM value

imFilenameCell = {['SPM00_TM??????_CM00_CHN01.klb'],['SPM00_TM??????_CM02_CHN00.klb'],['SPM00_TM??????_CM02_CHN04.klb'],['SPM00_TM??????_CM01_CHN05.klb'],['SPM00_TM??????_CM01_CHN07.klb'],['SPM00_TM??????_CM03_CHN06.klb'],['SPM00_TM??????_CM03_CHN02.klb'],['SPM00_TM??????_CM00_CHN03.klb']};


cameraTransformCell = [10, 21, 21, 30,30, 41, 41, 10];%CRITICAL. index indicating transformation selection based on flip and permutation to set all the views in the same x,y,z coordinate system. See function_multiview_camera_transformation for all the options.

TrCellPre = {[0 0 0],[88+5+1 186+27+3 15-11], [84+7 178+31 2],[2+1 142+10 22-5],[8 147 6],[70+7 183+15 -109-9],[74+5 176+28 -113-13],[0 0 0]};%Translation for each camera after the camera transform cell in order to perform coarse alignmnet. Leave empty if you want to automatically calculate it using normalized cross correlation

samplingXYZ = [0.40625, 0.40625, 1.625];%sampling in um

FWHMpsf = [0.8, 0.8, 4.0]; %theoretical full-width to half-max of the PSF in um.


outputFolderPattern = ['T:\temp\registration\diSPIM_20150824_220200\TM??????_localMaxImgBin\'];%outputfolder for debugging purposes. Leave empty for no debugging. This folder should be visible to the cluster if you want to run it on it

transposeOrigImage = false; %true if raw data was saved in tiff, since cluster PT transposed them after saving them in KLB.



%%
%RANSAC
%critical parameters for RANSAC alignment
RANSACparameterSet.minIntensityValue = 130; %global threshold. Any pixel below that intesnity will not be considered a point of interest for matching
RANSACparameterSet.blockSize = 128;         %blocks size (in pixels) around points of interest to match between views. The larger it is the more memory is required but the easier it is to match
RANSACparameterSet.searchRadius = 64;      %maximum distance (in pixels) between two views to match corresponding points after coarse alignment. If coarse alignment works well, this can be small. The smaller the value, the less memory is required.
RANSACparameterSet.thrPeakDOG = 25;         %CRITICAL: threshold to apply to Difference of Gaussians filtered image in order to find points of interest. 

RANSACparameterSet.interestPointDetector = 'localmaxima';        %select the interest point detector to find points to match between images. Options: 'DoG', 'localmaxima'

%usually "stable" parameters for RANSAC alignment
RANSACparameterSet.numHypothesis = 3;       %number of possible matches for each point of interest
RANSACparameterSet.thrNCC = 0.8;            %threshold of NCC to accept a match
RANSACparameterSet.numWorkers = -1;         %set to -1 to use as many as possible. If code runs out of memory, reduce the number.
RANSACparameterSet.maxNumPeaks = 150;       %maximum number of points of interest per view to match. The higher the number the longer the code takes
RANSACparameterSet.sigmaDOG = 3;          %sigma of the DoG to filter the image looking for points of interest  
RANSACparameterSet.thrMask = 130;           %global threshold to find "embryo mask" (so we can exclude beads from registration). Only needed for DoG detector

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
deconvParam.blockZsize = -1; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.

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
    function_multiview_coarse_fine_registration(imPath, imFilenameCellTM, cameraTransformCell, samplingXYZ, FWHMpsf, outputFolder, transposeOrigImage, RANSACparameterSet, deconvParam, TM, TrCellPre);
    
end