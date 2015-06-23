imPath = './testData/'
sigma_z = 1.0


%%
%generate simple test image
im = reshape([1:128*256*41],[128 256 41]);

for ii = 1:size(im,3)
    im(:,:,ii) = ii;
end


%%
%define transformation
A = [1      0.          0.          0;... 
    -0.     1           0.3         0; ...
    -0.     -0.3       sigma_z     0; ...
    0       0           0           1];
%find imSize limits
%imBounds = max(round(A(1:3,1:3) * size(im)'), size(im)');
imBounds = [];
%%
%apply imWarp
tic;

if( isempty(imBounds))
    imW = imwarp(im, affine3d(A), 'interp', 'cubic');
else
    imW = imwarp(im, affine3d(A), 'interp', 'cubic', 'Outputview', imref3d(imBounds'));
end
toc


%%
%apply our own imWarp
method = 3;
tic;
imF = imwarpfast(im, A, method,imBounds');
toc

%%

figure;imagesc(imW(:,:,30))
figure;imagesc(imF(:,:,30))

figure;
plot([squeeze(imF(24,24,:))';squeeze(imW(24,24,:))']');
legend('Ours','Warp');



err = max(abs(single(imW(:))-imF(:)))
if( err > 1e-3 )
    disp 'ERROR: TEST did not pass'
else
    disp 'OK. PASSED'
end

