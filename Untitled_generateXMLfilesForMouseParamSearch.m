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
FWHMpsfZvec = [3.0 5.0 7.0];
numItersVec = [20:20:80];
lambdaTVvec = [0 0.0001 0.001 ];


Nviews = length(imgFilenameCell);
PSFcell = cell(4,1);
psfFilenameCell = PSFcell;

count = 0;
for FWHMpsfZ = FWHMpsfZvec
    FWHMpsf = FWHMpsfOrig;
    FWHMpsf(3) = FWHMpsfZ;
    
    deconvParam.prefix = ['FWHMpsfZ_' num2str(round(FWHMpsfZ),'%.2d') ];
    
    %generate PSF files
    PSF = generatePSF(samplingXYZ, FWHMpsf, []);
    for ii = 1:Nviews
        %apply transformation
        PSFcell{ii} = single(imwarp(PSF, affine3d(Acell{ii}), 'interp', 'cubic'));
        
        %crop PSF to reduce it in size
        PSFcell{ii} = trimPSF(PSFcell{ii}, 1e-10);
        
        %save psf
        psfFilenameCell{ii} = [pathFiles 'psfFiles\psfReg_' deconvParam.prefix '_view' num2str(ii) '.klb'];        
        writeKLBstack(PSFcell{ii}, psfFilenameCell{ii}, -1, [],[],0,[]);
        psfFilenameCell{ii} = [pathFilesCluster 'psfFiles/psfReg_' deconvParam.prefix '_view' num2str(ii) '.klb'];        
    end
    
    for numIters = numItersVec
        deconvParam.numIter = numIters;
        
        for lambdaTV = lambdaTVvec
            deconvParam.lambdaTV = lambdaTV;                                    
            
            %save deconvolition XML parameters
            count = count + 1;
            filenameXML = [pathFiles 'xmlFiles\MVparam_' num2str(count) '.xml'];
            saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameClusterCell, psfFilenameCell, Acell, verbose, deconvParam);
        end
    end
end

