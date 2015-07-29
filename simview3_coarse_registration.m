%called from function_simview3_coarse_fine_registration.m
function imFilenameOutCell = simview3_coarse_registration(imPath, imFilenameCell, PSFcell, anisotropyZ, outputFolder, angles, numLevels, transposeOrigImage)

%%
%parameters
options.GPU = false;
options.Power2Flag = false;%memory consumption can be ridiculous


%%
%downsample PSF if we downsample images
for ii = 1:length(PSFcell)
   PSFcell{ii} = stackDownsample(PSFcell{ii}, numLevels); 
end

%%
%genarate transformations
tformCell = cell(length(angles),1);
imFilenameOutCell = cell(length(angles),1);
%calculate alignment for each view
for ii = 1:length(angles) %parfor here is not advisable because of memory usage
    %apply coarse transformation
    filename = [imPath imFilenameCell{ii}];
    im = readKLBstack(filename);
    
    if( transposeOrigImage )
       im = permute(im,[2 1 3]); 
    end
    
    if( ii == 1)
       imRefSize = ceil(size(im) .* [1 1 anisotropyZ]);
    end
    
    %flip and permutation and interpolation in z
    tformCell{ii} = coarseRegistrationBasedOnMicGeometry(im,angles(ii), anisotropyZ, imRefSize);
    
    %transform image
    %imTemp = imwarp(im, affine3d(tformCell{ii}), 'Outputview', imref3d(imRefSize), 'interp', 'linear');
    addpath './imWarpFast/'
    imTemp = imwarpfast(im, tformCell{ii}, 0, imRefSize);
    rmpath './imWarpFast/'
    
    %disp('===========DEBUGGING: no translation=============');
    %im = imTemp;  
    
    %downsample image
    imTemp = stackDownsample(imTemp, numLevels);
    
    %"double blur" image, so all images "look" the same
    if( mod(ii,2) == 0 )
        imTemp = uint16(convnfft(single(imTemp), single(PSFcell{ii}),'same',[1:max(ndims(imTemp),ndims(PSFcell{ii}))],options));
    else
        imTemp = uint16(convnfft(single(imTemp), single(PSFcell{ii}),'same',[1:max(ndims(imTemp),ndims(PSFcell{ii}))],options));
    end
    
    
    
    
    if( ii == 1 )
        numLevelsT = 2;
        imRefD = stackDownsample(imTemp,numLevelsT);%downsample to make it faster
        im = imTemp;
    else
        imD = stackDownsample(imTemp,2);%downsample to make it faster
        [Atr,nccVal] = fitTranslation(imRefD, imD, round(min(size(imD))/6));
        Atr(4,1:3) = Atr(4,1:3) * (2^numLevelsT);%to compensate for downsample
        tformCell{ii} = tformCell{ii} * Atr;
        
        %im = imwarp(im, affine3d(tformCell{ii}), 'Outputview', imref3d(imRefSize), 'interp', 'linear');
        addpath './imWarpFast/'
        im = imwarpfast(im, tformCell{ii}, 0, imRefSize);
        rmpath './imWarpFast/'
    end       
    
    %save image     
    imFilenameOutCell{ii} = ['imWarp_Matlab_CM' num2str(ii-1,'%.2d')];
    writeKLBstack(im, [outputFolder imFilenameOutCell{ii} '.klb']);
end
save([outputFolder 'imRegister_Matlab_tform.mat'],'tformCell','imPath','imFilenameCell', 'anisotropyZ', 'numLevels','imFilenameOutCell');


