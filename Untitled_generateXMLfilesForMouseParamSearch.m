%load transformation and image locations
load('mouse_15_04_03_TM000200_MVdeconv_param.mat');

%path to Janelia cluster to save files
pathFiles = 'B:\keller\deconvolution\mouse_15_04_03_TM200_test\';

pathFilesCluster = '/nobackup/keller/deconvolution/mouse_15_04_03_TM200_test/'
%invariant parameters for the test
samplingXYZ = [0.40625, 0.40625, 2.031];%sampling in um

FWHMpsfOrig = [0.8, 0.8, 3.0]; %theoretical full-width to half-max of the PSF in um.

deconvParam.verbose = 0; %set >0 to print out intermedate deconvolution steps for debugging
%deconvParam.lambdaTV = 0.0001; %Ttotal variation regularization lambda parameter. Set to < 0 to deactivate total variation regularization
%deconvParam.numIter = 40; %number of Lucy-Richardson iterations
deconvParam.imBackground = 100.0; %image background level. It will be subtracted from images to make sure noise statistics are as Poissonian as possible
deconvParam.blockZsize = 512; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.
%deconvParam.prefix = '';%we will use this for PSF differentiation

verbose = 0;

%%
%parameters to search
FWHMpsfZvec = [4.0];
numItersVec = [20:20:40];
lambdaTVvec = [0 ];


Nviews = length(imgFilenameCell);
psfFilenameCell = PSFcell;
psfFilenameClusterCell = PSFcell;
count = 36;
for FWHMpsfZ = FWHMpsfZvec
    FWHMpsf = FWHMpsfOrig;
    FWHMpsf(3) = FWHMpsfZ;
    
    deconvParam.prefix = ['FWHMpsfZ_' num2str(round(FWHMpsfZ),'%.2d') ];
    
    %generate PSF files    
    for ii = 1:Nviews        
        psfFilenameCell{ii} = [pathFiles 'psfFiles\psfReg_' deconvParam.prefix '_view' num2str(ii) '.klb'];                
        psfFilenameClusterCell{ii} = [pathFilesCluster 'psfFiles/psfReg_' deconvParam.prefix '_view' num2str(ii) '.klb'];        
    end
    PSFcell = generateTransformedPSF(samplingXYZ, FWHMpsf,Acell,psfFilenameCell);
    
    %generate xml
    for numIters = numItersVec
        deconvParam.numIter = numIters;
        
        for lambdaTV = lambdaTVvec
            deconvParam.lambdaTV = lambdaTV;                                    
            
            %save deconvolition XML parameters
            count = count + 1;
            filenameXML = [pathFiles 'xmlFiles\MVparam_' num2str(count) '.xml'];
            saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameClusterCell, psfFilenameClusterCell, Acell, verbose, deconvParam);
        end
    end
end

