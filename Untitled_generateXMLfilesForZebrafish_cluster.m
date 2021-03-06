TMvec = 190:620;

%path to Keller network for input files
pathInputFiles = 'S:\SiMView3\15-07-09\Dre_HuC_H2BGCaMP6s_0-1_20150709_170711.corrected\SPM00\TM??????\'
%path to Janelia cluster to save files
pathOutputFiles = 'B:\keller\deconvolution\Dre_HuC_20150709_170711.corrected\TM??????\';

pathFilesCluster = '/nobackup/keller/deconvolution/Dre_HuC_20150709_170711.corrected/TM??????/'

%invariant parameters for the test
samplingXYZ = [0.40625, 0.40625, 6.070];%in um
FWHMpsf = [0.8, 0.8, 5.0]; %theoretical full-width to half-max of the PSF in um.


deconvParam.lambdaTV = -1.0; %Ttotal variation regularization lambda parameter. Set to < 0 to deactivate total variation regularization
deconvParam.numIter = 40; %number of Lucy-Richardson iterations
deconvParam.imBackground = 100.0; %image background level. It will be subtracted from images to make sure noise statistics are as Poissonian as possible
deconvParam.blockZsize = -1; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.
deconvParam.prefix = '';%empty

verbose = 0;

%%
for TM = TMvec
    pathIn = recoverFilenameFromPattern(pathInputFiles,TM);
    pathOut = recoverFilenameFromPattern(pathOutputFiles,TM);
    pathCluster = recoverFilenameFromPattern(pathFilesCluster,TM);
    
    [AcellTM, imgFilenameCell] = readXMLdeconvolutionFile([pathIn 'MVrefine_deconv_LR_multiGPU_param_TM' num2str(TM,'%.6d') '.xml']);
    Acell = readXMLdeconvolutionFile([recoverFilenameFromPattern(pathInputFiles,400) 'MVrefine_deconv_LR_multiGPU_param_TM' num2str(400,'%.6d') '.xml']);%fixed time point to avoid jittering
    Nviews = length(Acell);
    
    imgFilenameClusterCell = cell(Nviews,1);
    psfFilenameClusterCell = cell(Nviews,1);    
    psfFilenameCell = cell(Nviews,1);
    
    %generate PSF files    
    for ii = 1:Nviews
       [PATHSTR,NAME,EXT] = fileparts(imgFilenameCell{ii}); 
       
       imgFilenameClusterCell{ii} = [pathCluster NAME EXT];
               
        %save psf
        psfFilenameCell{ii} = [pathOut NAME '_psfReg.klb'];                
        psfFilenameClusterCell{ii} = [pathCluster NAME '_psfReg.klb'];  
    end
    PSFcell = generateTransformedPSF(samplingXYZ, FWHMpsf,Acell,psfFilenameCell);
    
    %write out file
    filenameXML = [pathOut 'MVref_deconv_LR_multiGPU_param_JFCluster_TM' num2str(TM) '.xml'];
    saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameClusterCell, psfFilenameClusterCell, Acell, verbose, deconvParam);
end

