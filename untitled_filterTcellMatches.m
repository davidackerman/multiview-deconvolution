NCCthr = 0.7;

Tcell = TcellOrig;
%%
for ii = 1:size(Tcell,1)
    for jj = 1:size(Tcell,2)
        for kk = 1:length(Tcell{ii,jj})
            qq = Tcell{ii,jj}{kk};
            qq(qq(:,7) < NCCthr,:) = [];
            Tcell{ii,jj}{kk} = qq;
        end
    end
end


%%
%fit affine transformation for all views
[AcellRansac, statsCell] = fitAffineMultiviewRANSAC(Tcell, maxRadiusResidualInPixels, numTrialsRANSAC, numWorkers);

%%
[idxMaxInliers, idxMinAvgResidual] = parseRANSACstats(statsCell);
disp(['Avg. residual = ' num2str(mean(sqrt(sum(statsCell{idxMaxInliers}.residuals.^2,2)))) ' pixels for ' num2str(statsCell{idxMaxInliers}.numInliers) ' inliers'])

%collect final transform matrices
A = AcellRansac{idxMaxInliers};%final affines transformation
numViews = length(imFilename);
Acell = cell(numViews,1);
Acell{1} = eye(4);
for ii = 2:numViews
   Acell{ii} = [reshape(A(12 * (ii-2) + 1: 12 *(ii-1)),[4 3]), [0;0;0;1]]; 
   Acell{ii}
end

%%
%calculate plot fo rmathcing points
stats = statsCell{idxMaxInliers};
A = AcellRansac{idxMaxInliers};%final affines transformation
xyz = [];
for ii = 1:numViews-1

    H = stats.H(:, (ii-1) * 12 + 1 : ii * 12);
    Aaux = A((ii-1) * 12 + 1 : ii * 12);
    %aux = reshape(H * Aaux, [size(H,1)/3 3]);
    aux = H * Aaux;
    aux(aux < 0 ) = -1 * aux(aux < 0 );
    %xyz = [xyz; aux];
    
    
    xyz = [xyz; H(:,1:3)];
end

xyz = unique(xyz,'rows');

figure;
plot3(xyz(:,1), xyz(:,2), xyz(:,3),'o' );grid on;