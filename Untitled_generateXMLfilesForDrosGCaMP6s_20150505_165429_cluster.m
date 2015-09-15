TMvec = 2200:3250;

%path to Keller network for input files
pathInputFiles = 'S:\SiMView3\15-05-05\Dme_L1_57C10-GCaMP6s_20150505_165429.corrected\SPM00\TM??????\'

%path to Janelia cluster to save files
pathOutputFiles = 'B:\keller\deconvolution\Dme_L1_57C10-GCaMP6s_20150505_165429.corrected\SPM00\TM??????\';

pathFilesCluster = '/nobackup/keller/deconvolution/Dme_L1_57C10-GCaMP6s_20150505_165429.corrected/SPM00/TM??????/'


imgFilenameCellPattern = {['SPM00_TM??????_CM00_CHN01.klb'], ['SPM00_TM??????_CM02_CHN00.klb'], ['SPM00_TM??????_CM01_CHN01.klb'], ['SPM00_TM??????_CM03_CHN00.klb']};%filenames of each view. CRITICAL: the same order should be preserved when giving the transformations to match images between cameras


%invariant parameters for the test
samplingXYZ = [0.40625, 0.40625, 6.780];%in um
FWHMpsf = [0.8, 0.8, 4.0]; %theoretical full-width to half-max of the PSF in um.


deconvParam.lambdaTV = -1.0; %Ttotal variation regularization lambda parameter. Set to < 0 to deactivate total variation regularization
deconvParam.numIter = 20; %number of Lucy-Richardson iterations
deconvParam.imBackground = 100.0; %image background level. It will be subtracted from images to make sure noise statistics are as Poissonian as possible
deconvParam.blockZsize = -1; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.
deconvParam.prefix = '';%empty

verbose = 0;

%%
%load reference alignment file
tfine = load('T:\temp\deconvolution\20150505_165429_GCaMP6_TM001407\imWarp_Matlab_tform_fine');
tcoarse = load('T:\temp\deconvolution\20150505_165429_GCaMP6_TM001407\imRegister_Matlab_tform');

Nviews = length(tfine.tformCell);

Acell = cell(Nviews,1);
for ii = 1:Nviews
    Acell{ii} = tcoarse.tformCell{ii} * tfine.tformCell{ii};
end

%%
imgFilenameCell = cell(Nviews,1);
for TM = TMvec
    pathIn = recoverFilenameFromPattern(pathInputFiles,TM);
    pathOut = recoverFilenameFromPattern(pathOutputFiles,TM);
    pathCluster = recoverFilenameFromPattern(pathFilesCluster,TM);
    
    for ii = 1:Nviews
       imgFilenameCell{ii} = recoverFilenameFromPattern(imgFilenameCellPattern{ii},TM); 
    end
        
    
    imgFilenameClusterCell = cell(Nviews,1);
    psfFilenameClusterCell = cell(Nviews,1);    
    psfFilenameCell = cell(Nviews,1);
    psfFilenameCellKeller =  cell(Nviews,1);
    
    %generate PSF files    
    for ii = 1:Nviews
       [PATHSTR,NAME,EXT] = fileparts(imgFilenameCell{ii}); 
       
       imgFilenameClusterCell{ii} = [pathCluster NAME EXT];
       imgFilenameCell{ii} = [pathIn NAME EXT];
               
        %save psf
        psfFilenameCell{ii} = [pathOut NAME '_psfReg.klb'];                
        psfFilenameClusterCell{ii} = [pathCluster NAME '_psfReg.klb'];  
        psfFilenameCellKeller{ii} = [pathIn NAME '_psfReg.klb'];                
    end
    PSFcell = generateTransformedPSF(samplingXYZ, FWHMpsf,Acell,psfFilenameCell);
    PSFcell = generateTransformedPSF(samplingXYZ, FWHMpsf,Acell,psfFilenameCellKeller);
    
    %write out file for JFC clustar
    filenameXML = [pathOut 'MVref_deconv_LR_multiGPU_param_JFCluster_TM' num2str(TM) '.xml'];
    saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameClusterCell, psfFilenameClusterCell, Acell, verbose, deconvParam);
    
    %write out file for Keller network
    filenameXML = [pathIn 'MVref_deconv_LR_multiGPU_param_JFCluster_TM' num2str(TM) '.xml'];
    saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameCell, psfFilenameCellKeller, Acell, verbose, deconvParam);
    
end

