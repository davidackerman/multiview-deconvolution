%function test_fine_registration_blocks

%%
%parameters
minIntensityValue = 20;
blockSize = 16;
searchRadius = 16;

maxNumPeaks = 20;
sigmaDOG = 3.0;
thrPeakDOG = 2.0;


numHypothesis = 1;
numWorkers = 10;


numTrialsRANSAC = 100;
maxRadiusResidualInPixels = 10.0;


%%
%generate reference image
imRef = zeros([128,81, 55]); 
imRef = imRef + 10 + randn(size(imRef));%add a little bit of noise or normxcorr is unstable

%all in the same plane so it is easy to check results

imRef(50:70,30:31,10:13) = 90;
imRef(80:81,45:60,22:27) = 120;
imRef(25:30,20:25,22:27) = 100;
imRef(100:105,60:65,22:27) = 110;




%blur image
imRef = imgaussian(single(imRef), 2);

%generate transformations
A = cell(4,1);
for ii = 1:4
   A{ii} = eye(4); 
end
A{2}(4,1:3) = [12 -8  0];%translation
A{3}(4,1:3) = [-5 0  0];
A{4}(4,1:3) = [5 -7  1];

A{2}(1,2) = 0.2;
A{2}(2,1) = -0.2;%sligth rotation

%apply transformations
imCell = cell(4,1);
imCell{1} = imRef;
for ii = 2:4
    imCell{ii} = imwarp(imRef, affine3d(A{ii}), 'Outputview', imref3d(size(imRef)));
end

%%
%display images
figure;
for ii = 1:4
    subplot(2,2,ii);
    auxRGB = zeros(size(imRef,1),size(imRef,2), 3, 'uint8');
    auxRGB(:,:,1) = uint8(imCell{ii}(:,:,25));
    auxRGB(:,:,2) = uint8(imRef(:,:,25));
    imagesc(auxRGB);
    colormap gray
    title(['View ' num2str(ii)]);
end


%%
%find correspondences between images pairwise
numIm = length(imCell);
Tcell = cell(numIm, numIm);
for ii = 1:numIm
    imRef = imCell{ii};
    
    %detect points of interest in reference image
    interestPts = detectInterestPoints_DOG(imRef, sigmaDOG, maxNumPeaks, thrPeakDOG, imRef > minIntensityValue,0);
    
    %find correspondence for point of interest in the other images
    for jj = 1:numIm
        if( ii == jj )
            continue;
        end
        im = imCell{jj};
        Tcell{ii,jj} = pairwiseImageBlockMatching(imRef,im, blockSize, searchRadius, numHypothesis, interestPts(:,1:3), numWorkers);
    end
end

%%
[Acell, statsCell] = fitAffineMultiviewRANSAC(Tcell, maxRadiusResidualInPixels, numTrialsRANSAC, numWorkers);

%%
%select best RANSAC match
[idxMaxInliers, idxMinAvgResidual] = parseRANSACstats(statsCell);
disp(['Avg. residual = ' num2str(mean(sqrt(sum(statsCell{idxMaxInliers}.residuals.^2,2)))) ' pixels for ' num2str(statsCell{idxMaxInliers}.numInliers) ' inliers'])

%collect final transform matrices
AA = Acell{idxMaxInliers};%final affines transformation
numViews = length(imCell);
Acell = cell(numViews,1);
Acell{1} = eye(4);
for ii = 2:numViews
   Acell{ii} = [reshape(AA(12 * (ii-2) + 1: 12 *(ii-1)),[4 3]), [0;0;0;1]]; 
   Acell{ii}
   qq = fliptform(maketform('affine',A{ii}));
   qq.tdata.Tinv
end

%%
%apply perfect inverse
figure;
for ii = 1:4
    qq = fliptform(maketform('affine',A{ii}));
    
    aux = imwarp(imCell{ii}, affine3d(qq.tdata.Tinv), 'Outputview', imref3d(size(imRef)));
    if( ii == 1)
        auxRef = aux;
    end
    auxRGB = zeros(size(aux,1),size(aux,2), 3, 'uint8');
    auxRGB(:,:,1) = uint8(aux(:,:,25));
    auxRGB(:,:,2) = uint8(auxRef(:,:,25));
    subplot(2,2,ii);
    imagesc(auxRGB);
    colormap gray
    title(['View ' num2str(ii)]);
end


%%
%apply estimated inverse
figure;
for ii = 1:4    
    
    aux = imwarp(imCell{ii}, affine3d(Acell{ii}), 'Outputview', imref3d(size(imRef)));
    if( ii == 1)
        auxRef = aux;
    end
    auxRGB = zeros(size(aux,1),size(aux,2), 3, 'uint8');
    auxRGB(:,:,1) = uint8(aux(:,:,25));
    auxRGB(:,:,2) = uint8(auxRef(:,:,25));
    subplot(2,2,ii);
    imagesc(auxRGB);
    colormap gray
    title(['View ' num2str(ii)]);
end