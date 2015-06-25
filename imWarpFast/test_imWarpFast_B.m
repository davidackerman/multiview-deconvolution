imPath = './testData/'
sigma_z = 5.0


%%
%read original image
im = readTIFFstack([imPath 'drosophiila_origVol.tif']);



%%
%define transformation
A = [1      0.3          0.4          0;... 
    -0.3     1           0.         0; ...
    -0.     -0.       sigma_z     0; ...
    15       -7           4           1];
%find imSize limits
imBounds = max(round(A(1:3,1:3) * size(im)'), size(im)');

%%
%apply imWarp
tic;
imW = imwarp(im, affine3d(A), 'interp', 'cubic', 'Outputview', imref3d(imBounds'));
toc

writeKLBstack(imW, [imPath 'drosophila_imwarpMatlab.klb']);

%%
%apply our own imWarp
method = 3;
tic;
imF = imwarpfast(im, A, method,imBounds');
toc
writeKLBstack(imF, [imPath 'drosophila_imwarpFast.klb']);


%%

%err = max(abs(single(imW(:))-imF(:)))
qq = double(imW(:))-round(imF(:));
err = prctile(abs(qq(:)),[70 80 90 95 99])
if( err(4) > 1.0 )
    disp 'ERROR: TEST did not pass'
else
    disp 'OK. PASSED'
end

figure;imagesc(double(imW(:,:,30))-imF(:,:,30))
