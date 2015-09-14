
%%
%parameters
TMvec = [0:278]; %time points to be registered

imPathPattern = ['S:\SiMView1\15-08-10\Mmu_E1_mKate2_20150810_160708.corrected\']; %base folder where original images are located. ??? characters will be filled with the TM value

imFilenameCell = {['SPM00\TM??????\SPM00_TM??????_CM00_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM00_CHN00.klb'], ['SPM00\TM??????\SPM00_TM??????_CM01_CHN00.klb'], ['SPM01\TM??????\SPM01_TM??????_CM01_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras


disp('==============================================================================================================');
disp(['============YOU NEED TO HAVE AT LEAST ONE FILE NAMED ' imPathPattern 'MVrefine_deconv_LR_multiGPU_param_TM?????.xml CONTAINING A GOOD ALIGNMENT TO START THE PROCESS FOR THE OTHER TIME POINTS================']);
disp('==============================================================================================================');

samplingXYZ = [0.325, 0.325, 2.031];%sampling in um

FWHMpsf = [0.65, 0.65, 5.0]; %theoretical full-width to half-max of the PSF in um.

outputFolderPattern = [];%outputfolder for debugging purposes. Leave empty for no debugging. This folder should be visible to the cluster if you want to run it on it

transposeOrigImage = false; %true if raw data was saved in tiff, since cluster PT transposed them after saving them in KLB.


%%
%RANSAC
%critical parameters for RANSAC alignment> This is a refinement, so parameters can be tight
RANSACparameterSet.minIntensityValue = 150; %global threshold. Any pixel below that intesnity will not be considered a point of interest for matching
RANSACparameterSet.blockSize = 144;         %blocks size (in pixels) around points of interest to match between views. The larger it is the more memory is required but the easier it is to match
RANSACparameterSet.searchRadius = 64;      %maximum distance (in pixels) between two views to match corresponding points after coarse alignment. If coarse alignment works well, this can be small. The smaller the value, the less memory is required.

%usually "stable" parameters for RANSAC alignment
RANSACparameterSet.numHypothesis = 3;       %number of possible matches for each point of interest
RANSACparameterSet.thrNCC = 0.8;            %threshold of NCC to accept a match
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
deconvParam.lambdaTV = 0.0001; %Ttotal variation regularization lambda parameter. Set to < 0 to deactivate total variation regularization
deconvParam.numIter = 40; %number of Lucy-Richardson iterations
deconvParam.imBackground = 100.0; %image background level. It will be subtracted from images to make sure noise statistics are as Poissonian as possible
deconvParam.blockZsize = 512; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.

%%
isSubmitted = false(length(TMvec), 1);


while( isempty(isSubmitted == false) == false )
   
    for ii = 1:length(TMvec)
       if( isSubmitted(ii) == true )
           continue;
       end
       
       TM = TMvec(ii); 
       filenameXML = [imPathPattern 'MVrefine_deconv_LR_multiGPU_param_TM' num2str(TM,'%.6d') '.xml'];
       if( exist(filenameXML,'file') ~= 0 )
           isSubmitted(ii) = true; %file already exists for alignment (in case we restart the process)
           continue;
       end
        
       
       for jj = 1:4%size of the temporal window to look forward/backward. 4 is the minimum since we have 8 machines in the cluster so far
            filenameXML = [imPathPattern 'MVrefine_deconv_LR_multiGPU_param_TM' num2str(TM -jj,'%.6d') '.xml'];
            if( exist(filenameXML,'file') ~= 0 )                
                %make sure it was registered appropiately
               Acell = readXMLdeconvolutionFile(filenameXML);
               if( checkAffineTr(Acell) )
                break; 
               end
            end
            filenameXML = [imPathPattern 'MVrefine_deconv_LR_multiGPU_param_TM' num2str(TM +jj,'%.6d') '.xml'];
            if( exist(filenameXML,'file') ~= 0 )
                %make sure it was registered appropiately
                Acell = readXMLdeconvolutionFile(filenameXML);
                if( checkAffineTr(Acell) )
                    break;
                end
            end
            filenameXML = [];
       end
       
       if( isempty(filenameXML) == false ) %we have found a good initial condition
           %submit job to cluster
           
           outputFolder = [];
           if( isempty(outputFolderPattern) == false )
               outputFolder = recoverFilenameFromPattern(outputFolderPattern, TM);
               if( exist(outputFolder, 'dir') == 0 )
                   mkdir(outputFolder);
               end
           end
           
           imPath = recoverFilenameFromPattern(imPathPattern, TM);
           imFilenameCellTM = cell(length(imFilenameCell),1);
           for jj = 1:length(imFilenameCell)
               imFilenameCellTM{jj} = recoverFilenameFromPattern(imFilenameCell{jj},TM);              
           end
           
           %select number of workers
           if( RANSACparameterSet.numWorkers < 1 )
               qq = feature('numCores');
               RANSACparameterSet.numWorkers = qq;
           end
           RANSACparameterSet.numWorkers = min(12, RANSACparameterSet.numWorkers);%12 is the maximum allowed in current version
           
           
           %call registration function
           currentTime = clock;
           timeString = [...
               num2str(currentTime(1)) num2str(currentTime(2), '%.2d') num2str(currentTime(3), '%.2d') ...
               '_' num2str(currentTime(4), '%.2d') num2str(currentTime(5), '%.2d') num2str(round(currentTime(6) * 1000), '%.5d')];
           parameterDatabase = [pwd '\cluster_jobs\MV_ref_reg_jobParam_' timeString '.mat'];
           
           save(parameterDatabase, ...
                'imPath', 'imFilenameCellTM', 'Acell', 'samplingXYZ', 'FWHMpsf', 'outputFolder', 'transposeOrigImage', 'RANSACparameterSet', 'deconvParam', 'TM');
            %function to call
           %function_multiview_refine_registration_lowMem_cluster(parameterDatabase);
           cmdFunction = ['function_multiview_refine_registration_lowMem_cluster(''' parameterDatabase ''' )'];
           cmd = ['job submit /scheduler:keller-cluster.janelia.priv /user:simview /exclusive /nodegroup:Large_RAM " runMatlabJob.cmd """' pwd '""" """' cmdFunction '"""'];
           disp(['Suhmitting job for TM ' num2str(TM,'%.6d') ' with command ']);
           disp(cmd);
           [status, systemOutput] = system(cmd);
            
           %update vector
           isSubmitted(ii) = true;
       end
       
    end
    %pause for 60 seconds to check again submitted jobs
    pause(60);
end
