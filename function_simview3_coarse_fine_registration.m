%based on scripts script_simview3_fine_registration_blocks.m and
%script_simview3_coarse_registration.m
function function_simview3_coarse_fine_registration(imPath, imFilenameCell, samplingXYZ, FWHMpsfZ, outputFolder, transposeOrigImage, RANSACparameterSet)


%%
%fixed parameters
angles = [0 90 180 270];
numLevels = 0; %2^numLevels downsampling to perform this operations

anisotropyZ = samplingXYZ(3) / samplingXYZ(1);

%%
if( length(imFilenameCell) ~= 4 )
    error 'We need 4 images for angles 0,90,180,270 in order'
end

numViews = length(imFilenameCell);
%%
sampling = samplingXYZ(1) * ones(1,3);%in um (we make stacks isotropic using interpolation)
FWHMpsfOdd = [2*samplingXYZ(1), 2*samplingXYZ(1), FWHMpsfZ];%in um
FWHMpsfEven = [FWHMpsfZ, 2*samplingXYZ(1), 2*samplingXYZ(1)];%in um

PSFeven = generatePSF(sampling,FWHMpsfEven, []); 
PSFodd = generatePSF(sampling,FWHMpsfOdd, []); 

%generate PSF for the double blurring
PSFcell = cell(numViews,1);
for ii = 1:numViews
    if( mod(ii,2) == 0 )
        PSFcell{ii} = PSFeven;
    else
        PSFcell{ii} = PSFodd;
    end
end

%%
%coarse registration (basically flipdim + permute + scale
imFilenameOutCell = simview3_coarse_registration(imPath, imFilenameCell, PSFcell, anisotropyZ, outputFolder, angles, numLevels, transposeOrigImage);

%%
%fine registration

%fixed parameters
param.numHypothesis = 3;
param.numWorkers = min(12, feature('numCores'));
param.maxNumPeaks = 100;
param.sigmaDOG = 3.0 * 2;
param.thrMask = 110;% a little bit above background level

param.numTrialsRANSAC = 50000;
param.maxRadiusResidualInPixels = 15.0;
%--------------------------------------
switch(RANSACparameterSet)
    case 1%functional imaging               
        param.minIntensityValue = 150;
        param.blockSize = 144;%96;%critical to make sure NCC discriminates enough
        param.searchRadius = 64 * 2;        
        %it is better to find        
        param.thrPeakDOG = 15;
        
    case 2%nuclear marker development
        
        param.minIntensityValue = 150;
        param.blockSize = 144;%96;%critical to make sure NCC discriminates enough
        param.searchRadius = 64 * 2;
        
        %it is better to find
        param.thrPeakDOG = 18;
                
    case 3%membrane marker development
        param.minIntensityValue = 150;
        param.blockSize = 144;%96;%critical to make sure NCC discriminates enough
        param.searchRadius = 64 * 2;
        
        %it is better to find
        param.thrPeakDOG = 9;   
    case 4%functional imaging zebrafish
        param.minIntensityValue = 150;
        param.blockSize = 144;%96;%critical to make sure NCC discriminates enough
        param.searchRadius = 128;%param.blockSize/2 + 128;
        
        %it is better to find
        param.thrPeakDOG = 7;   
end
%%
simview3_fine_registration(outputFolder, imFilenameOutCell, param);