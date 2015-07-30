%called from function_simview3_coarse_fine_registration.m
function [imCell, tformCell] = function_multiview_coarse_registration(imPath, imFilenameCell, cameraTransformCell, PSF, anisotropyZ, outputFolder, transposeOrigImage)

%%
%parameters
options.GPU = false;
options.Power2Flag = false;%memory consumption can be ridiculous

Nviews = length(imFilenameCell);

PSFcell = cell(Nviews,1);

%%
%genarate transformations
tformCell = cell(Nviews,1);
imCell = cell(Nviews,1);
%calculate alignment for each view
for ii = 1:Nviews %parfor here is not advisable because of memory usage
    %apply coarse transformation
    disp(['Reading original image for view ' num2str(ii-1)]);
    tstart = tic;
    filename = [imPath imFilenameCell{ii}];
    imCell{ii} = readKLBstack(filename);
    disp(['Took ' num2str(toc(tstart)) ' secs']);
    
    if( transposeOrigImage )
       imCell{ii} = permute(imCell{ii},[2 1 3]); 
    end
    
    if( ii == 1)
       imRefSize = ceil(size(imCell{ii}) .* [1 1 anisotropyZ]);
    end
    
    disp(['Applying pre-defined flip and permutation to view ' num2str(ii-1)]);
    tstart = tic;
    %flip and permutation and interpolation in z to set all the views in
    %the same x,y,z coordinate system
    tformCell{ii} = function_multiview_camera_transformation(cameraTransformCell(ii), anisotropyZ, imRefSize);
    
    %transform image
    %imTemp = imwarp(imCell{ii}, affine3d(tformCell{ii}), 'Outputview', imref3d(imRefSize), 'interp', 'linear');
    addpath './imWarpFast/'
    imTemp = imwarpfast(imCell{ii}, tformCell{ii}, 0, imRefSize);
    rmpath './imWarpFast/'
    %transform PSF    
    PSFcell{ii} = imwarp(PSF, affine3d(tformCell{ii}), 'interp', 'cubic');
    disp(['Took ' num2str(toc(tstart)) ' secs']);
        
    disp(['Downsampling and calculating translation to align view ' num2str(ii-1) ' to reference view']);
    tstart = tic;
    %"double blur" image, so all images "look" the same
    if( mod(ii,2) == 0 )
        imTemp = uint16(convnfft(single(imTemp), single(PSFcell{ii}),'same',[1:max(ndims(imTemp),ndims(PSFcell{ii}))],options));
    else
        imTemp = uint16(convnfft(single(imTemp), single(PSFcell{ii}),'same',[1:max(ndims(imTemp),ndims(PSFcell{ii}))],options));
    end            
    
    if( ii == 1 )
        numLevelsT = 2;
        imRefD = stackDownsample(imTemp,numLevelsT);%downsample to make it faster
        imCell{ii} = imTemp;
    else
        imD = stackDownsample(imTemp,2);%downsample to make it faster
        [Atr,nccVal] = fitTranslation(imRefD, imD, round(min(size(imD))/6));
        Atr(4,1:3) = Atr(4,1:3) * (2^numLevelsT);%to compensate for downsample
        tformCell{ii} = tformCell{ii} * Atr;
        
        %imCell{ii} = imwarp(imCell{ii}, affine3d(tformCell{ii}), 'Outputview', imref3d(imRefSize), 'interp', 'linear');
        addpath './imWarpFast/'
        imCell{ii} = imwarpfast(imCell{ii}, tformCell{ii}, 0, imRefSize);
        rmpath './imWarpFast/'
    end       
    disp(['Took ' num2str(toc(tstart)) ' secs']);
    
    %save image     
    if( isempty(outputFolder) == false )
        disp(['Saving debugging information for view ' num2str(ii-1) ' in folder ' outputFolder]);
        tstart = tic;
        imFilenameOut = ['multiview_coarse_reg_view' num2str(ii-1,'%.2d')];
        imAux = single(imCell{ii});
        minI = min(imAux(:));
        maxI = max(imAux(:));
        imAux = uint16( 4096 * (imAux-minI) / (maxI-minI) );
        writeKLBstack(imAux, [outputFolder imFilenameOut '.klb']);
        disp(['Took ' num2str(toc(tstart)) ' secs']);
    end
end

if( isempty(outputFolder) == false )
    save([outputFolder 'multiview_coarse_reg.mat'],'tformCell','imPath','imFilenameCell', 'anisotropyZ');
end


