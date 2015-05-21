%combine coarse and fine registration to generate the final stack (using a
%single interpolation)

%pathImPattern = 'S:/SiMView3/15-04-24/Dme_L1_57C10-GCaMP6s_20150424_142342.corrected/SPM00/TM??????/SPM00_TM??????_'
function simview3_final_registration_and_deconvolution_timeSeries(pathImPattern, TM)

%%
%parameters
numItersLR = 20;

%%
%fixed parameters
baseRegistrationFolder = 'registrationFiles';
prefixRegistrationFiles = 'simview3_15_04_24_';

imSuffix = {'CM00_CHN01', 'CM02_CHN00', 'CM01_CHN01', 'CM03_CHN00'};
PSFfilename = 'PSF_synthetic.klb';

debugBasename = 'T:\temp\deconvolution\simview3_';%to save intermediate steps

%L-R options
lambdaTV = 0.008; %0.002 is value recommended by paper



%%
%load coarse registration
coarse = load([baseRegistrationFolder filesep prefixRegistrationFiles 'imRegister_Matlab_tform.mat'],'tformCell','imPath','imFilenameCell', 'numLevels', 'anisotropyZ');

%load fine registration
fine = load([baseRegistrationFolder filesep prefixRegistrationFiles 'imWarp_Matlab_tform_fine.mat'],'tformCell', 'Tcell', 'imPath', 'imFilename');

%%
%main loop to apply transformation to each image
Nviews = length(coarse.imFilenameCell);

imCell = cell(Nviews,1);
PSFcell = cell(Nviews,1);
weightsCell = cell(Nviews,1);

for ii = 1:Nviews
    %load image
    filename = recoverFilenameFromPattern(pathImPattern,TM);
    imFilename = [filename imSuffix{ii} '.klb'];
    disp(['Loading image ' imFilename]);
    imCell{ii} = readKLBstack(imFilename);
    
    %load PSF(we assume the same PSF for all views)
    filename = [baseRegistrationFolder filesep prefixRegistrationFiles PSFfilename];
    disp(['Loading PSF ' filename]);
    PSF = readKLBstack(filename);
    
    %--------------------------------
    %{
    disp '=============debug 1: downsample=========================='
    imCell{ii} = stackDownsample(imCell{ii}, 3);
    PSF = stackDownsample(PSF, 3);
    fine.tformCell{ii}(4,1:3) = fine.tformCell{ii}(4,1:3) / 8;
    coarse.tformCell{ii}(4,1:3) = coarse.tformCell{ii}(4,1:3) / 8;
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
        writeKLBstack(imCell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_imReg_' num2str(ii) '.klb']);
        writeKLBstack(PSFcell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_psfReg_' num2str(ii) '.klb']);
        writeKLBstack(weightsCell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_weightsReg_' num2str(ii) '.klb']);
    end
    
end

%%
%perform lucy richardson
disp 'Calculating multi-view deconvolution...'
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, numItersLR, lambdaTV, 0, [debugBasename 'TM' num2str(TM,'%6d') '_LR_iter']);

filename = recoverFilenameFromPattern(pathImPattern,TM);
LRfilename = [filename 'LRdeconv.klb'];
writeKLBstack(single(J), LRfilename);