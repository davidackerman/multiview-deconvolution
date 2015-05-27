NCCthr = 0.8;

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
end

%%
%calculate plot for mathcing points
stats = statsCell{idxMaxInliers};
A = AcellRansac{idxMaxInliers};%final affines transformation
xyz = [];

for ii = 1:length(stats.blockOffsetFinal)-1
    Haux = stats.H(stats.blockOffsetFinal(ii)+1:stats.blockOffsetFinal(ii+1),:);
    Baux = stats.B(stats.blockOffsetFinal(ii)+1:stats.blockOffsetFinal(ii+1));
    step = size(Haux,1) / 3;%order is [x_1, x_2,...,y_1, y_2, 
    
    for jj = 1:numViews-1
        H = Haux( :,(jj-1) * 12 + 1: jj*12);
        if( sum(H(:)) < 0 )
            H = -H;
        end
        aux = H * A((jj-1) * 12 + 1: jj*12);
        
        if( sum(abs(aux)) > 0 )
           xyz = [xyz; reshape(aux, [length(aux)/3, 3])]; 
        end
    end
end

xyz = unique(xyz,'rows');

figure;
plot3(xyz(:,1), xyz(:,2), xyz(:,3),'o' );grid on;

%%
%overlay with reference image
im = readKLBstack('E:\simview3_deconvolution\15_04_24_fly_functionalImage\TM1445\Matlab_coarse_fine_register_blockRANSAC_full_resolution\imWarp_Matlab_CM00_regRANSAC.klb');
zz = unique(round(xyz(:,3)));
for ii = 1:length(zz)
   pp = (find(round(xyz(:,3)) == zz(ii)));
   
   figure;
   imagesc(im(:,:,zz(ii)));
   colormap gray;
   title(num2str(zz(ii)));
   hold on;
   plot(xyz(pp,1),xyz(pp,2),'go');
end