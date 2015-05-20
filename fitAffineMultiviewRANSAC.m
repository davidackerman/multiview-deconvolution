function [Acell, statsCell] = fitAffineMultiviewRANSAC(Tcell, maxRadiusResidualInPixels, numIters, numWorkers)

Acell = cell(numIters,1);
statsCell = Acell;

if( matlabpool('size') ~= numWorkers )
   
    if( matlabpool('size') > 0 )
        matlabpool('close');
    end
    matlabpool(numWorkers);
end


%disp '===================WARNING: parfor disconnected for debugging purposes=========================='
parfor ii = 1:numIters
   [Acell{ii}, statsCell{ii}] = fitAffineMultiviewRANSACiteration(Tcell, maxRadiusResidualInPixels); 
end



