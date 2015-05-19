%blockSize is recommended to be a power of 2 for speed
function Tcell = pairwiseImageBlockMatching(imRef,im, blockSize, searchRadius, numHypothesis, interestPts, numWorkers)

N = size(interestPts,1);
Tcell = cell(N,1);


if( matlabpool('size') ~= numWorkers )
   
    if( matlabpool('size') > 0 )
        matlabpool('close');
    end
    matlabpool(numWorkers);
end

interestPts = round(interestPts);

%disp '===================WARNING: parfor disconnected for debugging purposes=========================='
parfor count = 1:N
    
    ii = interestPts(count,1);
    jj = interestPts(count,2);
    kk = interestPts(count,3);
    
    %crop blocks
    
    imBlock = im( max(ii - blockSize/2 + 1,1) : min(ii + blockSize/2, size(im,1)), max(jj - blockSize/2 + 1,1) : min(jj + blockSize/2, size(im,2)),max(kk - blockSize/2 + 1,1) : min(kk + blockSize/2, size(im,3)));
    imRefblock = imRef( max(ii - blockSize/2 + 1 - searchRadius,1) : min(ii + blockSize/2 + searchRadius, size(im,1)), max(jj - blockSize/2 + 1 - searchRadius,1) : min(jj + blockSize/2 + searchRadius, size(im,2)),max(kk - blockSize/2 + 1 - searchRadius,1) : min(kk + blockSize/2 + searchRadius, size(im,3)));
            
    %calculate registration
    Taux = imRegistrationTranslationMultipleHypothesis(imRefblock, imBlock, numHypothesis);
    
    xyz = repmat([(ii-1) * blockSize + 1, (jj-1) * blockSize + 1, (kk-1) * blockSize + 1],[numHypothesis 1]);
    Tcell{count} = [xyz, xyz + Taux(:,1:3), Taux(:,4)];
    
    
end