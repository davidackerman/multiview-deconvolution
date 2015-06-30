%combine coarse and fine registration to generate the final stack (using a
%single interpolation)

%pathImPattern = 'S:/SiMView3/15-04-24/Dme_L1_57C10-GCaMP6s_20150424_142342.corrected/SPM00/TM??????/SPM00_TM??????_'
function function_simview3_final_registration_and_deconvolution(pathImPattern, TM, numItersLR, backgroundOffset, baseRegistrationFolder, PSFfilename, transposeOrigImage)

%%
%parameters
%numItersLR = 40;
%backgroundOffset = 100;

%%
%fixed parameters
%baseRegistrationFolder = 'registrationFiles';
imSuffix = {'CM00_CHN01', 'CM02_CHN00', 'CM01_CHN01', 'CM03_CHN00'};
%PSFfilename = 'PSF_synthetic.klb';

debugBasename = [baseRegistrationFolder filesep 'simview3_'];%to save intermediate steps

%L-R options
lambdaTV = 0.008; %0.002 is value recommended by paper



%%
%load coarse registration
coarse = load([baseRegistrationFolder filesep  'imRegister_Matlab_tform.mat'],'tformCell','imPath','imFilenameCell', 'numLevels', 'anisotropyZ');

%load fine registration
fine = load([baseRegistrationFolder filesep  'imWarp_Matlab_tform_fine.mat'],'tformCell', 'Tcell', 'imPath', 'imFilename');

%%
%main loop to apply transformation to each image
Nviews = length(coarse.imFilenameCell);

imCell = cell(Nviews,1);
PSFcell = imCell;
weightsCell = imCell;

for ii = 1:Nviews
    %load image
    filename = recoverFilenameFromPattern(pathImPattern,TM);
    imFilename = [filename imSuffix{ii} '.klb'];
    disp(['Loading image ' imFilename]);
    imCell{ii} = readKLBstack(imFilename);
    
    if( transposeOrigImage )
       imCell{ii} = permute(imCell{ii},[2 1 3]); 
    end
    
    %load PSF(we assume the same PSF for all views)
    filename = [baseRegistrationFolder filesep  PSFfilename];
    disp(['Loading PSF ' filename]);
    PSFcell{ii} = readKLBstack(filename);
    
    %--------------------------------
    %{
    disp '=============debug 1: downsample=========================='
    imCell{ii} = stackDownsample(imCell{ii}, 3);
    PSFcell{ii} = stackDownsample(PSFcell{ii}, 3);
    fine.tformCell{ii}(4,1:3) = fine.tformCell{ii}(4,1:3) / 8;
    coarse.tformCell{ii}(4,1:3) = coarse.tformCell{ii}(4,1:3) / 8;
    %}
    %--------------------------------------
    
    if( ii == 1 )%calculate reference side only once
        imSizeRef = ceil(size(imCell{ii}) .* [1 1 coarse.anisotropyZ]);
    end
    
    %calculate contrast weights
    disp('Calculating contrast weights');
    weightsCell{ii} = single(estimateDeconvolutionWeights(imCell{ii}, coarse.anisotropyZ , 15, []));
        
    
    %calculate transform    
    A = coarse.tformCell{ii} * fine.tformCell{ii};
    
    %apply transformation
    disp 'Applying transformation to image, PSF and weights'
    addpath './imWarpFast'    
    imCell{ii} = imwarpfast(imCell{ii},A, 2, imSizeRef);
    weightsCell{ii} = imwarpfast(weightsCell{ii},A, 0, imSizeRef);
% %     PSFcell{ii} = imwarpfast(PSFcell{ii},A, 3, []); %code is not working right now without imSizeRef
    rmpath './imWarpFast'
% %     imCell{ii} = imwarp(imCell{ii}, affine3d(A), 'Outputview', imref3d(imSizeRef), 'interp', 'cubic', 'FillValues', backgroundOffset);%to avoid "edge" effect
% %     weightsCell{ii} = imwarp(weightsCell{ii}, affine3d(A), 'Outputview', imref3d(imSizeRef), 'interp', 'linear');
    PSFcell{ii} = imwarp(PSFcell{ii}, affine3d(A), 'interp', 'cubic');
    
    %save results if requested        
    if( ~isempty(debugBasename) )
        disp(['Writing registered files to ' debugBasename '*Reg_' num2str(ii) '.klb']);
        writeKLBstack(imCell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_imReg_' num2str(ii) '.klb']);
        writeKLBstack(PSFcell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_psfReg_' num2str(ii) '.klb']);
        writeKLBstack(weightsCell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_weightsReg_' num2str(ii) '.klb']);
        
% %         writeRawStack(imCell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_imReg_' num2str(ii) '.raw']);
% %         writeRawStack(PSFcell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_psfReg_' num2str(ii) '.raw']);
% %         writeRawStack(weightsCell{ii},[debugBasename 'TM' num2str(TM,'%6d') '_weightsReg_' num2str(ii) '.raw']);
    end
    
end


%%
%perform lucy richardson
disp 'Calculating multi-view deconvolution...'
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, backgroundOffset, numItersLR, lambdaTV, 0, [debugBasename 'TM' num2str(TM,'%6d') '_LR_iter']);

filename = recoverFilenameFromPattern(pathImPattern,TM);
LRfilename = [filename 'LRdeconv.klb'];
writeKLBstack(single(J), LRfilename);