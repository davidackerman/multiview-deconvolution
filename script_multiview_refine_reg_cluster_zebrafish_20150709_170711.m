%%
%parameters

TMvec = [190:620]; %time points to be registered

imPathPattern = ['S:\SiMView3\15-07-09\Dre_HuC_H2BGCaMP6s_0-1_20150709_170711.corrected\SPM00\TM??????\']; %base folder where original images are located. ??? characters will be filled with the TM value

imFilenameCell = {['SPM00_TM??????_CM00_CHN01.klb'], ['SPM00_TM??????_CM02_CHN00.klb'], ['SPM00_TM??????_CM01_CHN01.klb'], ['SPM00_TM??????_CM03_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras


%affine transformation usually read from XML file
Acell = { [1.00000000,0.00000000,0.00000000,0.00000000,0.00000000,1.00000000,0.00000000,0.00000000,0.00000000,0.00000000,14.94153846,0.00000000,0.00000000,0.00000000,0.00000000,1.00000000],...
      [0.99639221,0.01282788,-0.13997657,98.82690996,0.00755412,-0.00295258,15.08454007,5.45410981,-0.01198898,-1.04654861,0.49816563,953.07858899,0.00000000,0.00000000,0.00000000,1.00000000],...
      [0.98147414,0.01414121,-0.17017858,39.86720956,0.00686813,-1.00914091,0.10632789,1016.40106235,0.00114313,-0.00174388,14.97805795,-3.57060421,0.00000000,0.00000000,0.00000000,1.00000000],...
      [0.99540513,-0.02410037,-0.32241773,80.71205542,0.00895985,-0.00139534,15.00095940,5.51518447,-0.00485881,1.03753587,0.15998882,61.17493591,0.00000000,0.00000000,0.00000000,1.00000000]};


samplingXYZ = [0.40625, 0.40625, 6.070];%in um

FWHMpsf = [0.8, 0.8, 5.0]; %theoretical full-width to half-max of the PSF in um.

%outputFolderPattern = ['T:\temp\registration\zebrafish_20150709_170711\refReg_TM??????\'];%outputfolder for debugging purposes. Leave empty for no debugging. This folder should be visible to the cluster if you want to run it on it
outputFolderPattern = [];

transposeOrigImage = false; %true if raw data was saved in tiff, since cluster PT transposed them after saving them in KLB.

useCluster = true; %set to true if you want to run in Keller cluster

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
deconvParam.blockZsize = -1; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.

%%
%parameters
for TM = TMvec
        
    %%
    outputFolder = recoverFilenameFromPattern(outputFolderPattern, TM);
    if( isempty(outputFolder) == false && exist(outputFolder, 'dir') == 0 )
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
    
    %call registration function (local)
    if( useCluster == false )
        function_multiview_refine_registration(imPath, imFilenameCellTM, Acell, samplingXYZ, FWHMpsf, outputFolder, transposeOrigImage, RANSACparameterSet, deconvParam, TM);
    else
        %call registration function with job submit
        currentTime = clock;
        timeString = [...
            num2str(currentTime(1)) num2str(currentTime(2), '%.2d') num2str(currentTime(3), '%.2d') ...
            '_' num2str(currentTime(4), '%.2d') num2str(currentTime(5), '%.2d') num2str(round(currentTime(6) * 1000), '%.5d')];
        parameterDatabase = [pwd '\cluster_jobs\MV_ref_reg_jobParam_' timeString '.mat'];
        
        save(parameterDatabase, ...
            'imPath', 'imFilenameCellTM', 'Acell', 'samplingXYZ', 'FWHMpsf', 'outputFolder', 'transposeOrigImage', 'RANSACparameterSet', 'deconvParam', 'TM');
        %function to call
        %function_multiview_refine_registration_lowMem_cluster(parameterDatabase);
        cmdFunction = ['function_multiview_refine_registration(''' parameterDatabase ''' )'];
        cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /exclusive " runMatlabJob.cmd """' pwd '""" """' cmdFunction '"""'];
        disp(['Suhmitting job for TM ' num2str(TM,'%.6d') ' with command ']);
        disp(cmd);
        [status, systemOutput] = system(cmd);
    end
end