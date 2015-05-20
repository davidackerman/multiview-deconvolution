%function test_fitAffineMultiviewRANSACiteration

numViews = 4;
T = [[ 0 0 0];...
     [-4 10 1];...%one translation per view
     [5 -1 0];...
     [20 3.2 10.2]];

maxRadiusResidualInPixels = 5.0;
numHypothesis = 5;

%%
%number of points per view
numPts = randi(50, [numViews 1]) + 10;

%generate synthetic Tcell with only translation
TcellSyn = cell(numViews);

for ii = 1:numViews
    for jj = 1:numViews
        if( ii == jj )
            continue;
        end
        
        nn = numPts(ii);
        
        isOutlier = rand(numHypothesis, nn) < 0.2; 
        
        pts = 100 * rand(nn,3);        
        TcellSyn{ii,jj} = cell(nn,1);
        for kk = 1:nn
            TcellSyn{ii,jj}{kk} = [repmat([pts(kk,:) + T(ii,:), pts(kk,:)+T(jj,:)],[numHypothesis 1] ) , rand(numHypothesis,1)];
        end
    end
end

%%
[A, stats] = fitAffineMultiviewRANSACiteration(TcellSyn, maxRadiusResidualInPixels);

%%
%collect matrices
Acell = cell(numViews,1);
Acell{1} = eye(4);
for ii = 2:numViews
   Acell{ii} = [reshape(A(12 * (ii-2) + 1: 12 *(ii-1)),[4 3]), [0;0;0;1]]; 
   Acell{ii}
end

T
