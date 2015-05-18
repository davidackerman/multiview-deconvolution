%blockSize is recommended to be a power of 2 for speed
function Tcell = pairwiseImageBlockMatching(imRef,im, blockSize, searchRadius, numHypothesis, minIntensityValue, numWorkers)


numBlocks = [numel(1:blockSize:size(im,1)), numel(1:blockSize:size(im,2)), numel(1:blockSize:size(im,3))];
Tcell = cell(prod(numBlocks),1);


if( matlabpool('size') ~= numWorkers )
   
    if( matlabpool('size') > 0 )
        matlabpool('close');
    end
    matlabpool(numWorkers);
end

disp '===================WARNING: parfor disconnected for debugging purposes=========================='
for count = 1:prod(numBlocks)
    %for ii = 1:blockSize:size(im,1)
    %    for jj = 1:blockSize:size(im,2)
    %        for kk = 1:blockSize:size(im,3)
    
    
    %calculate indexes
    countAux = count - 1;
    kk = mod(countAux,numBlocks(3));
    countAux = (countAux - kk) / numBlocks(3);
    jj = mod(countAux, numBlocks(2));
    countAux = (countAux - jj) / numBlocks(2);
    ii = countAux;
    
    %matlab indexes
    kk = kk + 1;
    jj = jj + 1;
    ii = ii + 1;
    
    %crop blocks
    imBlock = im((ii-1) * blockSize + 1: min(ii*blockSize,size(im,1)), (jj-1) * blockSize + 1: min(jj*blockSize,size(im,2)), (kk-1) * blockSize + 1: min(kk*blockSize,size(im,3)));
    imRefblock = imRef(max((ii-1) * blockSize + 1 - searchRadius,1): min(ii*blockSize + searchRadius,size(im,1)), max((jj-1) * blockSize + 1 - searchRadius, 1): min(jj*blockSize + searchRadius,size(im,2)), max((kk-1) * blockSize + 1 - searchRadius,1): min(kk*blockSize + searchRadius,size(im,3)));
    
    if( max(imBlock(:)) < minIntensityValue )%no information in this 
        continue;
    end
    
    %calculate registration
    Taux = imRegistrationTranslationMultipleHypothesis(imRefblock, imBlock, numHypothesis);
    
    xyz = repmat([(ii-1) * blockSize + 1, (jj-1) * blockSize + 1, (kk-1) * blockSize + 1],[numHypothesis 1]);
    Tcell{count} = [xyz, xyz + Taux(:,1:3), Taux(:,4)];
    %        end
    %    end
    %end        
    
end