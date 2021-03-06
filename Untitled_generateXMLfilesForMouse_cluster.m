TMvec = 20:333;

%path to Keller network for input files
pathInputFiles = 'S:\SiMView1\15-04-03\Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked\'
%path to Janelia cluster to save files
pathOutputFiles = 'B:\keller\deconvolution\Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked\SPM00\TM??????\';

pathFilesCluster = {'/nobackup/keller/deconvolution/Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked/SPM00/TM??????/', '/nobackup/keller/deconvolution/Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked/SPM01/TM??????/'};

%invariant parameters for the test
samplingXYZ = [0.40625, 0.40625, 2.031];%in um
FWHMpsf = [0.8, 0.8, 5.0]; %theoretical full-width to half-max of the PSF in um.


deconvParam.lambdaTV = -1.0; %Ttotal variation regularization lambda parameter. Set to < 0 to deactivate total variation regularization
deconvParam.numIter = 40; %number of Lucy-Richardson iterations
deconvParam.imBackground = 100.0; %image background level. It will be subtracted from images to make sure noise statistics are as Poissonian as possible
deconvParam.blockZsize = 512; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.
deconvParam.prefix = '';%empty

verbose = 0;

%%
for TM = TMvec
    pathIn = recoverFilenameFromPattern(pathInputFiles,TM);
    pathOut = recoverFilenameFromPattern(pathOutputFiles,TM);
    pathCluster = cell(1,2);
    pathCluster{1} = recoverFilenameFromPattern(pathFilesCluster{1},TM);
    pathCluster{2} = recoverFilenameFromPattern(pathFilesCluster{2},TM);
    
    [Acell, imgFilenameCell] = readXMLdeconvolutionFile([pathIn 'MVrefine_deconv_LR_multiGPU_param_TM' num2str(TM,'%.6d') '.xml']);
    Nviews = length(Acell);
    
    imgFilenameClusterCell = cell(Nviews,1);
    psfFilenameClusterCell = cell(Nviews,1);
    psfFilenameCell = cell(Nviews,1);
    %generate PSF files    
    for ii = 1:Nviews
       [PATHSTR,NAME,EXT] = fileparts(imgFilenameCell{ii}); 
       
       if( ii == 1 || ii == 3)
        imgFilenameClusterCell{ii} = [pathCluster{1} NAME EXT];
       else
           imgFilenameClusterCell{ii} = [pathCluster{2} NAME EXT];
       end       
        %save psf
        psfFilenameCell{ii} = [pathOut NAME '_psfReg.klb'];                
        psfFilenameClusterCell{ii} = [pathCluster{1} NAME '_psfReg.klb'];  
    end
    PSFcell = generateTransformedPSF(samplingXYZ, FWHMpsf,Acell,psfFilenameCell);
    
    %write out file
    filenameXML = [pathOut 'MVref_deconv_LR_multiGPU_param_JFCluster_TM' num2str(TM) '.xml'];
    saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameClusterCell, psfFilenameClusterCell, Acell, verbose, deconvParam);
end

