%called from function_simview3_coarse_fine_registration.m
function [imCell, tformCell] = function_multiview_coarse_registration(imPath, imFilenameCell, cameraTransformCell, PSF, anisotropyZ, outputFolder, transposeOrigImage, TrCellPre)

%%
%parameters
options.GPU = false;
options.Power2Flag = false;%memory consumption can be ridiculous
numLevelsT = 2;

%%
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
       
       %transform PSF: we need before hand to blur other views
       for jj = 1:Nviews
           %flip and permutation and interpolation in z to set all the views in
           %the same x,y,z coordinate system
           tformCell{jj} = function_multiview_camera_transformation(cameraTransformCell(jj), anisotropyZ, imRefSize);
           PSFcell{jj} = single(imwarp(PSF, affine3d(tformCell{jj}), 'interp', 'cubic'));
           %downsample (we only need it for matching translation) and it was very slow at full resolution
           PSFcell{jj} = stackDownsample(PSFcell{jj},numLevelsT);
           PSFcell{jj} = PSFcell{jj} / sum(PSFcell{jj}(:));
       end
    end
    
    disp(['Applying pre-defined flip and permutation to view ' num2str(ii-1)]);
    tstart = tic;    
    
    
    if( isempty( TrCellPre ) )
        calculateTr = true;
        %transform image
        %imTemp = imwarp(imCell{ii}, affine3d(tformCell{ii}), 'Outputview', imref3d(imRefSize), 'interp', 'linear');
        addpath './imWarpFast/'
        imTemp = imwarpfast(imCell{ii}, tformCell{ii}, 1, imRefSize);
        rmpath './imWarpFast/'
        disp(['Took ' num2str(toc(tstart)) ' secs']);
        
        %save image before appling translation
        if( isempty(outputFolder) == false )
            disp(['Saving debugging information for view ' num2str(ii-1) ' in folder ' outputFolder]);
            tstart = tic;
            imFilenameOut = ['multiview_predefined_reg_view' num2str(ii-1,'%.2d')];
            imAux = single(imTemp);
            minI = min(imAux(:));
            maxI = max(imAux(:));
            imAux = uint16( 4096 * (imAux-minI) / (maxI-minI) );
            writeKLBstack(imAux, [outputFolder imFilenameOut '.klb']);
            disp(['Took ' num2str(toc(tstart)) ' secs']);
            clear imAux;
        end
    else %apply the flips + especified translation
        calculateTr = false;
        %transform image
        Tr = eye(4);
        Tr(4,1:3) = TrCellPre{ii};
        tform = tformCell{ii} * Tr;
        
        addpath './imWarpFast/'
        imCell{ii} = imwarpfast(imCell{ii}, tform, 1, imRefSize);
        rmpath './imWarpFast/'
        tformCell{ii} = tform;
        disp(['Took ' num2str(toc(tstart)) ' secs']);
    end    
    
    if( calculateTr )
        disp(['Downsampling, blurring and calculating translation to align view ' num2str(ii-1) ' to reference view']);
        tstart = tic;
        if( ii == 1 )
            imRefD = stackBinning(imTemp,numLevelsT);%downsample to make it faster
            %"double blur" image, so all images "look" the same
            for jj = 1:Nviews
                if( ii ~= jj)
                    imRefD = convnfft(single(imRefD), single(PSFcell{ii}),'same',[1:max(ndims(imRefD),ndims(PSFcell{ii}))],options);
                end
            end
        else
            imD = stackBinning(imTemp,numLevelsT);%downsample to make it faster
            %"double blur" image, so all images "look" the same
            for jj = 1:Nviews
                if( ii ~= jj)
                    imD = convnfft(single(imD), single(PSFcell{ii}),'same',[1:max(ndims(imD),ndims(PSFcell{ii}))],options);
                end
            end
            [Atr,nccVal] = fitTranslation(imRefD, imD, round(min(size(imD))/6));
            Atr(4,1:3) = Atr(4,1:3) * (2^numLevelsT);%to compensate for downsample
            tformCell{ii} = tformCell{ii} * Atr;
        end
        
        %imCell{ii} = imwarp(imCell{ii}, affine3d(tformCell{ii}), 'Outputview', imref3d(imRefSize), 'interp', 'linear');
        addpath './imWarpFast/'
        imCell{ii} = imwarpfast(imCell{ii}, tformCell{ii}, 1, imRefSize);%we transform the original image
        rmpath './imWarpFast/'
        
        disp(['Took ' num2str(toc(tstart)) ' secs']);
    end
    
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


