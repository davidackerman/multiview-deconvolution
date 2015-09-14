
%pathImPattern = 'S:/SiMView3/15-04-24/Dme_L1_57C10-GCaMP6s_20150424_142342.corrected/SPM00/TM??????/SPM00_TM??????_'
function function_simview3_apply_final_registration(pathImPattern, TM, baseRegistrationFolder, PSFfilename, transposeOrigImage, anisotropyZ)

%%
%fixed parameters
%baseRegistrationFolder = 'registrationFiles';
imSuffix = {'CM00_CHN01', 'CM02_CHN00', 'CM01_CHN01', 'CM03_CHN00'};
%PSFfilename = 'PSF_synthetic.klb';

debugBasename = [baseRegistrationFolder filesep 'simview3_'];%to save intermediate steps


%%
%load PSF(we assume the same PSF for all views)
disp(['Loading PSF ' PSFfilename]);
PSForig = readKLBstack(PSFfilename);

%%
%main loop to apply transformation to each image
Nviews = length(imSuffix);
addpath './imWarpFast'
for ii = 1:Nviews
    
    
    
    %load transforms
    A = load([baseRegistrationFolder filesep 'affineTr_view' num2str(ii) '.txt']);
    
    %load image
    filename = recoverFilenameFromPattern(pathImPattern,TM);
    imFilename = [filename imSuffix{ii} '.klb'];
    disp(['Loading image ' imFilename]);
    im = readKLBstack(imFilename);
    
    if( transposeOrigImage )
        im = permute(im,[2 1 3]);
    end   
    
    
    if( ii == 1 )%calculate reference side only once
        imSizeRef = ceil(size(im) .* [1 1 anisotropyZ]);
    end
    
    %calculate contrast weights
    disp('Calculating contrast weights');
    weights = single(estimateDeconvolutionWeights(im, anisotropyZ , 15, []));
    
    
    %apply transformation
    disp 'Applying transformation to image, PSF and weights'
    
    im = imwarpfast(im,A, 2, imSizeRef);
    weights = imwarpfast(weights,A, 0, imSizeRef);
    PSF = imwarp(PSForig, affine3d(A), 'interp', 'cubic');
    
    %save results
    
    disp(['Writing registered files to ' debugBasename '*Reg_' num2str(ii) '.klb']);
    writeKLBstack(single(im),[debugBasename 'TM' num2str(TM,'%6d') '_imReg_' num2str(ii) '.klb']);
    writeKLBstack(single(PSF),[debugBasename 'TM' num2str(TM,'%6d') '_psfReg_' num2str(ii) '.klb']);
    writeKLBstack(single(weights),[debugBasename 'TM' num2str(TM,'%6d') '_weightsReg_' num2str(ii) '.klb']);   

end
rmpath './imWarpFast'

