function [idxMaxInliers, idxMinAvgResidual] = parseRANSACstats(RANSACstatsCell)

idxMaxInliers = -1;
maxInliners = -1;

idxMinAvgResidual = -1;
minAvgResidual = inf;

for ii = 1:length(RANSACstatsCell)
    
    stats = RANSACstatsCell{ii};
    
    %num inliers
    if( stats.numInliers > maxInliners )
        maxInliners = stats.numInliers;
        idxMaxInliers = ii;
    end
    
    %residual
    rr = mean( sqrt(sum(stats.residuals.^2,2)) );
    
    if( stats.numInliers > 0 && rr < minAvgResidual )
        minAvgResidual = rr;
        idxMinAvgResidual = ii;
    end
end
