%combine coarse and fine registration to generate the final stack (using a
%single interpolation)
%function script_simview3_final_registration()

%%
%parameters

%%
%fixed parameters
baseRegistrationFolder = 'D:\simview3_deconvolution\15_04_24_fly_functionalImage\TM1445\Matlab_coarse_fine_register_blockRANSAC_full_resolution'

PSFfilename = 'PSF_synthetic.klb'

debugBasename = 'D:\temp\deconvolution\simview3_';%to save intermediate steps

%L-R options
lambdaTV = 0.008; %0.002 is value recommended by paper
numItersLR = 30;


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
    disp(['Loading image ' filename]);
    imCell{ii} = readKLBstack(filename);
    
    %load PSF(we assume the same PSF for all views)
    disp(['Loading PSF ' baseRegistrationFolder filesep PSFfilename]);
    PSF = readKLBstack([baseRegistrationFolder filesep PSFfilename]);
    
    %--------------------------------
    %{
    disp '=============debug 1: downsample=========================='
    imCell{ii} = stackDownsample(imCell{ii}, 2);
    PSF = stackDownsample(PSF, 2);
    fine.tformCell{ii}(4,1:3) = fine.tformCell{ii}(4,1:3) / 4;
    coarse.tformCell{ii}(4,1:3) = coarse.tformCell{ii}(4,1:3) / 4;
    %}
    %--------------------------------------
    
    if( ii == 1 )%calculate reference side only once
        imSizeRef = ceil(size(imCell{1}) .* [1 1 coarse.anisotropyZ]);
    end
    
    %calculate contrast weights
    disp('Calculating contrast weights');
    weightsCell{ii} = single(estimateDeconvolutionWeights(imCell{ii}, coarse.anisotropyZ , 15, []));
        
    
    %calculate transform    
    A = coarse.tformCell{ii} * fine.tformCell{ii};
    
    %apply transformation
    disp 'Applying transformation to image, PSF and weights'
    imCell{ii} = imwarp(imCell{ii}, affine3d(A), 'Outputview', imref3d(imSizeRef), 'interp', 'cubic');
    weightsCell{ii} = imwarp(weightsCell{ii}, affine3d(A), 'Outputview', imref3d(imSizeRef), 'interp', 'linear');
    PSFcell{ii} = imwarp(PSF, affine3d(A), 'interp', 'cubic');
    
    %save results if requested    
    if( ~isempty(debugBasename) )
        disp(['Writing registered files to ' debugBasename '*Reg_' num2str(ii) '.klb']);
        writeKLBstack(imCell{ii},[debugBasename 'imReg_' num2str(ii) '.klb']);
        writeKLBstack(PSFcell{ii},[debugBasename 'psfReg_' num2str(ii) '.klb']);
        writeKLBstack(weightsCell{ii},[debugBasename 'weightsReg_' num2str(ii) '.klb']);
    end
    
end

%%
%perform lucy richardson
disp 'Calculating multi-view deconvolution...'
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, numItersLR, lambdaTV, 0, [debugBasename 'LR_iter']);

[pathstr,name,ext] = fileparts(coarse.imFilenameCell{1});
writeKLBstack(single(J), [baseRegistrationFolder filesep name 'LR_deconvolution.klb']);