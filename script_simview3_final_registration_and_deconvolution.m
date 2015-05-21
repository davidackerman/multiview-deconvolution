%combine coarse and fine registration to generate the final stack (using a
%single interpolation)
%function script_simview3_final_registration()

%%
%parameters

%%
%fixed parameters
baseRegistrationFolder = 'E:\simview3_deconvolution\15_04_24_fly_functionalImage\TM1445\Matlab_coarse_register_downsample2x_doubleBlurred'

PSFfilename = 'TODO.klb'

debugBasename = 'E:\temp\deconvolution\simview3_';%to save intermediate steps

%L-R options
lambdaTV = 0.008; %0.002 is value recommended by paper
numItersLR = 20;


%%
%load coarse registration
coarse = load([baseRegistrationFolder '\imRegister_Matlab_tform.mat'],'tformCell','imPath','imFilenameCell', 'numLevels', 'anisotropyZ');

%load fine registration
fine = load([baseRegistrationFolder '\imWarp_Matlab_tform_fine.mat'],'tformCell', 'Tcell', 'imPath', 'imFilename');

%%
%main loop to apply transformation to each image
Nviews = length(coarse.imFilenameCell);

imCell = cell(Nviews,1);
PSFcell = cell(Nviews,1);
weightsCell = cell(Nviews,1);

for ii = 1:Nviews
    %load image
    filename = [coarse.imPath filesep coarse.imFilenameCell{ii}];
    imCell{ii} = readKLBstack(filename);
    
    disp '=============debug 1=========================='
    imCell(ii) = stackDownsample(imCell{ii}, 2);
    
    imSizeRef = ceil(size(imCell{1}) .* [1 1 coarse.anisotropyZ]);
    
    %calculate contrast weights
    single(estimateDeconvolutionWeights(imCell{ii}, coarse.anisotropyZ , 15, []));
    
    %load PSF(we assume the same PSF for all views)
    PSF = readKLBstack(PSFfilename);
    
    %calculate transform
    disp '====================TODO:check with scaled data if this is correct================='
    A = fine.tformCell{ii} * coarse.tformCell{ii};
    
    %apply transformation
    imCell{ii} = imwarp(imCell{ii}, affine3d(A), 'Outputview', imSizeRef, 'interp', 'cubic');
    weightsCell{ii} = imwarp(weightsCell{ii}, affine3d(A), 'Outputview', imSizeRef, 'interp', 'linear');
    PSFcell{ii} = imwarp(PSF, affine3d(A), 'interp', 'cubic');
    
    %save results if requested
    if( ~isempty(debugBasename) )
        writeKLBstack(imCell{ii},[debugBasename 'imReg_' num2str(ii) '.klb']);
        writeKLBstack(psfCell{ii},[debugBasename 'psfReg_' num2str(ii) '.klb']);
        writeKLBstack(weightsCell{ii},[debugBasename 'weightsReg_' num2str(ii) '.klb']);
    end
    
end

%%
%perform lucy richardson
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, numItersLR, lambdaTV, 0, debugBasename);