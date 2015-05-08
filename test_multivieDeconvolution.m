%from http://www.mathworks.com/help/images/examples/deblurring-images-using-the-lucy-richardson-algorithm.html

numViews = 2;
sigma = [1 2;...
         3 1];
     
sigmaNoise = 0;

numIters = 20;

lambdaTV = 0.008; %0.002 is value recommended by paper

%%
%read original clean image
I = imread('board.tif');
I = rgb2gray( I(50+(1:256),2+(1:256),:) );
figure;
imagesc(I);
title('Original Image');

%%
%simulated blur
PSFcell = cell(numViews,1);
imCell = PSFcell;
for ii = 1:numViews
    v = fspecial('gaussian', [6*sigma(ii,1), 1], sigma(ii,1) ); % vertical filter
    h = fspecial('gaussian', [1, 6*sigma(ii,2)], sigma(ii,2) ); % horizontal
    PSFcell{ii} = v*h;
    imCell{ii} = single(convn(I,PSFcell{ii},'same')) + sigmaNoise * randn(size(I,1),size(I,2),'single');    
    figure;
    imagesc( imCell{ii});
    title(['Blurred view ' num2str(ii)]);
end

%%
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, numIters, lambdaTV, 1, './temp/test_');

%%
%run regular lucy richardson for one view
J1 = deconvlucy(imCell{1},PSFcell{1}, numIters);
figure;
imagesc(J1);
title('Single view Lucy-Richardson');
