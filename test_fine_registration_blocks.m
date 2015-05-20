%function test_fine_registration_blocks

%%
%parameters
minIntensityValue = 20;
blockSize = 16;
searchRadius = 32;

maxNumPeaks = 20;
sigmaDOG = 3.0;
thrPeakDOG = 2.0;


numHypothesis = 5;
numWorkers = 10;

%%
%generate reference image
imRef = zeros([128,81, 55]); 
imRef = imRef + 10 + randn(size(imRef));%add a little bit of noise or normxcorr is unstable

%all in the same plane so it is easy to check results
imRef(30:34,14:18,22:27) = 100;
imRef(90:92,30:38,22:27) = 50;
imRef(80:85,50:53,22:27) = 40;
imRef(90:92,30:38,22:27) = 60;
imRef(100:107,60:68,22:27) = 70;
imRef(50:52,20:22,22:27) = 80;
imRef(60:65,30:38,22:27) = 90;

%blur image
imRef = imgaussian(single(imRef), 2);

%generate transformations
A = cell(4,1);
for ii = 1:4
   A{ii} = eye(4); 
end
A{2}(4,1:3) = [10 -3  0];%translation
A{3}(4,1:3) = [-20 0  0];
A{4}(4,1:3) = [5 -7  1];

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
    imagesc(imCell{ii}(:,:,25));
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
    for jj = ii+1:numIm
        im = imCell{jj};
        Tcell{ii,jj} = pairwiseImageBlockMatching(imRef,im, blockSize, searchRadius, numHypothesis, interestPts(:,1:3), numWorkers);
    end
end

%%
%tformCell = fitAffineMultiview(Tcell);