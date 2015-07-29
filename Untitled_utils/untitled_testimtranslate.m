im = zeros([30 32 24]);
im(7:9,20:22,10:11) = 100;

T = [-2 3 5];

tt = [ 1 0 0 0;  0 0 -1 0; 0 -1 0 0; 0 size(im,1) size(im,3) 1];
%tt = eye(4);

im2 = imwarp(im, affine3d(tt), 'Outputview', imref3d(size(im)), 'interp', 'linear');

imA = imtranslate(im2, T, 0, 'linear', 1);

A = [[eye(3) [0;0;0]];[T([2 1 3]) 1]];

%A = tt * A;

A = tt;
A(4,1:3) = A(4,1:3) + T([2 1 3]);

imB = imwarp(im, affine3d(A), 'Outputview', imref3d(size(imA)), 'interp', 'linear');

max(imA(:))
norm(imA(:)-im(:))
norm(imA(:)-imB(:))