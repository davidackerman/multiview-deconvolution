%parameters

TM = [1]; %time points to be registered
imPathPattern = ['S:\SiMView3\15-08-24\Dme_E1_His2AvRFP_01234567_diSPIM_20150824_220200.corrected\SPM00\TM??????\']; %base folder where original images are located. ??? characters will be filled with the TM value
imFilenameCell = {['SPM00_TM??????_CM00_CHN01.klb'],['SPM00_TM??????_CM02_CHN00.klb'],['SPM00_TM??????_CM02_CHN04.klb'],['SPM00_TM??????_CM01_CHN05.klb'],['SPM00_TM??????_CM01_CHN07.klb'],['SPM00_TM??????_CM03_CHN06.klb'],['SPM00_TM??????_CM03_CHN02.klb'],['SPM00_TM??????_CM00_CHN03.klb']};

samplingXYZ = [0.40625, 0.40625, 1.625];%sampling in um

FWHMpsf = [0.8, 0.8, 4.0]; %theoretical full-width to half-max of the PSF in um.

weigthPower = 1.0;
weigthThr = 0.4;

sigmaGaussian = 10;%in pixels: smooth decay between regions with high contrast

%%
Nviews = length(imFilenameCell);
anisotropyZ = samplingXYZ(3) / samplingXYZ(1);
imFilenameCellTM = cell(length(imFilenameCell),1);
for ii = 1:length(imFilenameCell)
    imFilenameCellTM{ii} = recoverFilenameFromPattern([imPathPattern imFilenameCell{ii}],TM);
end

%%
parfor ii = 1:Nviews
   im = readKLBstack( imFilenameCellTM{ii} );
   
   ww = single(estimateDeconvolutionWeights(single(im), anisotropyZ , 15, []));
   
   ww = ww.^weigthPower;
   
   BW = ww > weigthThr;
   %BWorig = BW;
   %do it slice by slice
   kernelSize = ceil(6 *sigmaGaussian);
   neigh = ones(kernelSize);
   mask = zeros(size(BW));
   for jj = 1:size(BW,3)
       BW(:,:,jj) = imdilate(BW(:,:,jj),neigh);
       mask(:,:,jj) = imgaussian(single(BW(:,:,jj)),sigmaGaussian, kernelSize);
   end
  
   %write out new image
   filename = [imFilenameCellTM{ii}(1:end-4) '_masked.klb'];
   writeKLBstack(single(single(im) .* mask), filename);
end
