%test PSF transform without specifiying boundaries and with affine
%transformation using flip

imPath = './testData/'

%%
%load data
data = load([imPath 'PSFtest.mat']);

%%
%generate simple test image
im = data.im;

%%
%define transformation
A = data.A;

%find imSize limits
imBounds = [];

%%
%apply imWarp without bounding box
imW = imwarp(im, affine3d(A), 'interp', 'cubic');

%%
%apply our own imWarp
method = 3;
tic;
imF = imwarpfast(im, A, method,imBounds);
toc

%%
figure;imagesc(imW(:,:,30))
figure;imagesc(imF(:,:,30))

err = max(abs(single(imW(:))-imF(:)))
if( err > 1e-3 )
    disp 'ERROR: TEST did not pass'
else
    disp 'OK. PASSED'
end

