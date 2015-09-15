%parameters

TM = [1]; %time points to be registered
imPathPattern = ['S:\SiMView3\15-08-24\Dme_E1_His2AvRFP_01234567_diSPIM_20150824_220200.corrected\SPM00\TM??????\']; %base folder where original images are located. ??? characters will be filled with the TM value
imFilenameCell = {['SPM00_TM??????_CM00_CHN01.klb'],['SPM00_TM??????_CM02_CHN00.klb'],['SPM00_TM??????_CM02_CHN04.klb'],['SPM00_TM??????_CM01_CHN05.klb'],['SPM00_TM??????_CM01_CHN07.klb'],['SPM00_TM??????_CM03_CHN06.klb'],['SPM00_TM??????_CM03_CHN02.klb'],['SPM00_TM??????_CM00_CHN03.klb']};

samplingXYZ = [0.40625, 0.40625, 1.625];%sampling in um

FWHMpsf = [0.8, 0.8, 4.0]; %theoretical full-width to half-max of the PSF in um.


cameraTransformCell = [10, 21, 21, 30,30, 41, 41, 10];%CRITICAL. index indicating transformation selection based on flip and permutation to set all the views in the same x,y,z coordinate system. See function_multiview_camera_transformation for all the options.

imgBackground = 100;%we need to subtract before applying the mask
imgThreshold = 110;%for masking
sigmaGaussian = 30;%in pixels: smooth decay between regions with high contrast

minSigmaSmooth = 20;%so decay can start even before the half

%%
Nviews = length(imFilenameCell);
anisotropyZ = samplingXYZ(3) / samplingXYZ(1);
imFilenameCellTM = cell(length(imFilenameCell),1);
for ii = 1:length(imFilenameCell)
    imFilenameCellTM{ii} = recoverFilenameFromPattern([imPathPattern imFilenameCell{ii}],TM);
end

%%
imCell = cell(Nviews,1);
maskCell = cell(Nviews,1);
parfor ii = 1:Nviews
    tic;
    im = readKLBstack( imFilenameCellTM{ii} );
    
    maskHard = maskEmbryo(im, imgThreshold);
    
    %{
    intXY = zeros(size(im));
    for jj = 1:size(im,3)
       intXY(:,:,jj) = cumsum(mask(:,:,jj),1);
    end
   
    %do the same along YZ slices
    intYZ = zeros(size(im));
    for jj = 1:size(im,1)
       intYZ(jj,:,:) = cumsum(squeeze(mask(jj,:,:)),2) * anisotropyZ;
    end
    %}
    
    %perform PCA
    idx = find(maskHard > 0 );
    [x,y,z] = ind2sub(size(maskHard),idx);
    p0 = mean([x y z]);%center
    
    %in this case the axis pretty much align with x,y,z
    %xyz = bsxfun(@minus,[x y z],p0);
    %[U,S,V] = svd(xyz,'econ');
    
    
    %define 4 quadrants from center
    qI = zeros(4,1);
    
    qI(1) = prctile(im(idx( x > p0(1) & z > p0(3) )), 90);
    qI(2) = prctile(im(idx( x > p0(1) & z <= p0(3) )), 90);
    qI(3) = prctile(im(idx( x <= p0(1) & z > p0(3) )), 90);
    qI(4) = prctile(im(idx( x <= p0(1) & z <= p0(3) )), 90);
    
    [~,pos] = max(qI);
    
    BW = false(size(im));
    switch(pos)
        case 1
            BW(idx( x > p0(1) & z > p0(3))) = true;
        case 2
            BW(idx( x > p0(1) & z <= p0(3))) = true;
        case 3
            BW(idx( x <= p0(1) & z > p0(3))) = true;
        case 4
            BW(idx( x <= p0(1) & z <= p0(3))) = true;
    end
    
    
    %better to smooth in 3D
    kernelSize = ceil(6 *sigmaGaussian);
    if( kernelSize > 1 )
        neigh = ones(kernelSize, kernelSize, round(kernelSize / anisotropyZ));
        BW = imdilate(BW,neigh);
    end
    sigma = max(minSigmaSmooth, sigmaGaussian);
    kernelSize = ceil(6 *sigma);
    mask = imgaussianAnisotropy(single(BW),[sigma, sigma, sigma/anisotropyZ], [kernelSize, kernelSize, round(kernelSize / anisotropyZ)]);
    %{
   for jj = 1:size(BW,3)
       BW(:,:,jj) = imdilate(BW(:,:,jj),neigh);
       mask(:,:,jj) = imgaussian(single(BW(:,:,jj)),sigmaGaussian, kernelSize);
   end
    %}
    
    
   %generate masked image
   imMasked = single(im) - imgBackground;
   imMasked = imMasked .* mask;
   imMasked = imMasked + imgBackground;
   %write out new image
   filename = [imFilenameCellTM{ii}(1:end-4) '_decay.klb'];
   writeKLBstack(single(imMasked), filename);
   filename = [imFilenameCellTM{ii}(1:end-4) '_decay_mask.klb'];
   writeKLBstack(uint8(mask * 255), filename);
    
    maskCell{ii} = mask;
    imCell{ii} = im;
    toc;
end
%{
%%
%normalize masks by averaging them (but only in an area around the center)
%we use the camera transforma cell to divide the masks in two groups
groupIdx = cell(2,1);
for ii = 1:Nviews
    if( mod( floor(cameraTransformCell(ii) / 10) ,2 ) == 1 )
        groupIdx{1} = [groupIdx{1} ii];
    else
        groupIdx{2} = [groupIdx{2} ii];
    end
end

for gg = 1:2
    maskAvg = zeros(size(maskCell{1}));
    for ii = groupIdx{gg}
        maskAvg = maskAvg + maskCell{ii};
    end
    maskAvg( maskAvg == 0 ) = 1.0;%to avoid NaN
    
    for ii = groupIdx{gg}
        maskCell{ii} = maskCell{ii} ./ maskAvg;
        
    end
end

for ii = 1:Nviews
    im = imCell{ii};
    mask = maskCell{ii};
    imMasked = single(im) - imgBackground;
    imMasked = imMasked .* mask;
    imMasked = imMasked + imgBackground;
    %write out new image
    filename = [imFilenameCellTM{ii}(1:end-4) '_decay.klb'];
    writeKLBstack(single(imMasked), filename);
    filename = [imFilenameCellTM{ii}(1:end-4) '_decay_mask.klb'];
    writeKLBstack(uint8(mask * 255), filename);
end
%}


