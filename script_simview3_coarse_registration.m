
imPath = 'S:\SiMView3\15-04-24\Dme_L1_57C10-GCaMP6s_20150424_142342.corrected\SPM00\TM001445\'
imFilenameCell = {'SPM00_TM001445_CM00_CHN01.klb', 'SPM00_TM001445_CM02_CHN00.klb', 'SPM00_TM001445_CM01_CHN01.klb', 'SPM00_TM001445_CM03_CHN00.klb'};%0,90,180,270 degrees
anisotropyZ = 5.2 / 0.406;

outputFolder = 'E:\simview3_deconvolution\15_04_24_fly_functionalImage\TM1445\'

%%
%fixed parameters
angles = [0 90 180 270];

sampling = 0.40625 * ones(1,3);%in um
FWHMpsfOdd = [0.8 0.8 6.0];%in um
FWHMpsfEven = [6.0 0.8 0.8];%in um

numLevels = 0; %2^numLevels downsampling to perform this operations


options.GPU = false;
options.Power2Flag = false;%memory consumption can be ridiculous

%%
%generate PSF
PSFeven = generatePSF(sampling,FWHMpsfEven, []); 
PSFodd = generatePSF(sampling,FWHMpsfOdd, []); 

%%
%genarate transformation
tformCell = cell(length(angles),1);

%prepare image reference (0 degrees angles)
filename = [imPath imFilenameCell{1}];
imRef = readKLBstack(filename);
[A,imRef] = coarseRegistrationBasedOnMicGeometry(imRef,0, anisotropyZ);
tformCell{1} = A;

%"double blur" image, so all images "look" the same
imRef = convnfft(imRef, PSFeven,'same',[1:max(ndims(PSFeven),ndims(imRef))],options);

%downsample image
imRef = stackDownsample(imRef, numLevels);

%save image reference
writeKLBstack(imRef, [outputFolder 'imRegister_Matlab_CM' num2str(0,'%.2d') '.klb']);

%save image reference
disp '=====================debugging 0========================='
im = readKLBstack(filename);
im = imwarp(im, affine3d(tformCell{1}), 'Outputview', imref3d(size(imRef)), 'interp', 'linear');
writeKLBstack(im, [outputFolder 'imWarp_Matlab_CM' num2str(0,'%.2d') '.klb']);

%%
%calculate alignment for each view
for ii = 2:length(angles) %parfor here is not advisable because of memory usage
    %apply coarse transformation
    filename = [imPath imFilenameCell{ii}];
    im = readKLBstack(filename);
    
    %flip and permutation and interpolation in z
    [A, im] = coarseRegistrationBasedOnMicGeometry(im,angles(ii), anisotropyZ, size(imRef));
            
    %"double blur" image, so all images "look" the same
    if( mod(ii,2) == 0 )
        im = convnfft(im, PSFeven,'same',[1:max(ndims(im),ndims(PSFeven))],options);
    else
        im = convnfft(im, PSFodd,'same',[1:max(ndims(im),ndims(PSFodd))],options);
    end
    
    %downsample image
    im = stackDownsample(im, numLevels);
    
    %find translation    
    [T, im] = imRegistrationTranslationFFT(imRef, im);
    
    %generate affine matrix
    A(4,1:3) = A(4,1:3) + T([2 1 3]);%imwarp permutes x y wrt to imtranslate
    tformCell{ii} = A;
    
    
    %adjust image to imRef size
    if( size(imRef,1) <= size(im,1) )
        im = im(1:size(imRef,1),:,:);
    else
        im = padarray(im,[size(imRef,1)-size(im,1) 0 0],0,'post');
    end
    if( size(imRef,2) <= size(im,2) )
        im = im(:,1:size(imRef,2),:);
    else
        im = padarray(im,[0 size(imRef,2)-size(im,2) 0],0,'post');
    end
    if( size(imRef,3) <= size(im,3) )
        im = im(:,:, 1:size(imRef,3));
    else
        im = padarray(im,[0 0 size(imRef,3)-size(im,3)],0,'post');
    end
    
    
    %save image 
    writeKLBstack(im, [outputFolder 'imRegister_Matlab_CM' num2str(ii-1,'%.2d') '.klb']);
    
    %save image using imwarp
    disp '=====================debugging 2========================='
    im = readKLBstack(filename);
    im = imwarp(im, affine3d(tformCell{ii}), 'Outputview', imref3d(size(imRef)), 'interp', 'linear');
    writeKLBstack(im, [outputFolder 'imWarp_Matlab_CM' num2str(ii-1,'%.2d') '.klb']);
end
save([outputFolder 'imRegister_Matlab_tform.mat'],'tformCell','imPath','imFilenameCell', 'anisotropyZ', 'numLevels');

%%
%fine registration
%look at scriot_simview3_fine_registration_block.m