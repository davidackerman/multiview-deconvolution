%blockSize is recommended to be a power of 2 for speed
function Tcell = pairwiseImageBlockMatching(imRef,im, blockSize, searchRadius, numHypothesis, interestPts, numWorkers, thrNCC)

N = size(interestPts,1);
Tcell = cell(N,1);

if( exist('thrNCC','var') == 0 )
    thrNCC = 0.8;
end

if( matlabpool('size') ~= numWorkers )
   
    if( matlabpool('size') > 0 )
        matlabpool('close');
    end
    matlabpool(numWorkers);
end

interestPts = round(interestPts);



%%
%partition the for loop into chunks so I avoid taking so much memory
%(parfor is ridiculus)
offsetN = 1;
batchSize = max(numWorkers, 36);
while( offsetN <= N )
    %%
    %precompute patches so parfor does not copy big image to all the workers
    imBlockCell = cell(batchSize,1);
    imRefBlockCell = cell(batchSize,1);
    offsetVec = zeros(batchSize,3);
    countLocal = 1;
    Nlocal = 0;
    for count = offsetN:min(offsetN + batchSize-1, N);
        ii = interestPts(count,1);
        jj = interestPts(count,2);
        kk = interestPts(count,3);
        
        imBlockCell{countLocal} = single(im( max(ii - blockSize/2 + 1,1) : min(ii + blockSize/2, size(im,1)), max(jj - blockSize/2 + 1,1) : min(jj + blockSize/2, size(im,2)),max(kk - blockSize/2 + 1,1) : min(kk + blockSize/2, size(im,3))));
        imRefBlockCell{countLocal} = single(imRef( max(ii - blockSize/2 + 1 - searchRadius,1) : min(ii + blockSize/2 + searchRadius, size(im,1)), max(jj - blockSize/2 + 1 - searchRadius,1) : min(jj + blockSize/2 + searchRadius, size(im,2)),max(kk - blockSize/2 + 1 - searchRadius,1) : min(kk + blockSize/2 + searchRadius, size(im,3))));
        
        %account for offset from creating blocks (element might not be
        %centered)
        offsetVec(countLocal,:) = 0.5 * [max(ii - blockSize/2 + 1,1) + min(ii + blockSize/2, size(im,1)), max(jj - blockSize/2 + 1,1) + min(jj + blockSize/2, size(im,2)),max(kk - blockSize/2 + 1,1) + min(kk + blockSize/2, size(im,3))] -...
            0.5 * [max(ii - blockSize/2 + 1 - searchRadius,1) + min(ii + blockSize/2 + searchRadius, size(im,1)), max(jj - blockSize/2 + 1 - searchRadius,1) + min(jj + blockSize/2 + searchRadius, size(im,2)),max(kk - blockSize/2 + 1 - searchRadius,1) + min(kk + blockSize/2 + searchRadius, size(im,3))];
        
        countLocal = countLocal + 1;
        Nlocal = Nlocal + 1;
    end
    
    %%
    %disp '===================WARNING: parfor disconnected for debugging purposes=========================='
    TcellAux = cell(Nlocal,1);
    parfor count = 1:Nlocal
        %calculate registration
        Taux = imRegistrationTranslationMultipleHypothesis(imRefBlockCell{count}, imBlockCell{count}, numHypothesis, thrNCC);        
        if( ~isempty(Taux) )
            
            Taux(:, 1:3) = Taux(:, 1:3) + repmat(offsetVec(count,:), [size(Taux,1) 1]);
            
            xyz = repmat(interestPts(count + offsetN - 1,[2 1 3]),[size(Taux,1) 1]);
            TcellAux{count} = [xyz, xyz - Taux(:,[2 1 3]), Taux(:,4)];%to match imwarp when fitting affine
        end
    end
    %copy results
    for count = 1:Nlocal
        Tcell{count + offsetN - 1} = TcellAux{count};
    end
    %%
    %update offset count
    offsetN = offsetN + batchSize;
end