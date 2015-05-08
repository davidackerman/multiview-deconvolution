% im: 3D image with data from light microscopy
% imW: float 3D weights for each pixel (array is the same size as im)
% blockSize: window size to perform DCT and estimate shaprness at each
% point

function imW = estimateDeconvolutionWeights(im, anisotropyZ, cellDiameterPixels, blockSize)

if( isempty(blockSize) )
    blockSize = 8;
end


%%
%estimate sharpness per block and per slice
imW = zeros(size(im));
for ii = 1:size(im,3)%no parfor because function is already parallelized   
    imW(:,:,ii) = estimateSliceSharpnessDCT(im(:,:,ii), blockSize);    
end

%%
%smooth out results in 3D (we smooth about the size of a cell to avoid
%artifacts)
sigma = 0.5 * [cellDiameterPixels, cellDiameterPixels, cellDiameterPixels / anisotropyZ];

imW = imgaussianAnisotropy(single(imW),sigma);

%normalize
imW = imW - min(imW(:));
imW = imW / max(imW(:));
