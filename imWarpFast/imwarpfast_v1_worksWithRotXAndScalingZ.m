%  mode: If 0: linear interpolation and outside pixels set to nearest pixel
%            1: linear interpolation and outside pixels set to zero
%            2: cubic interpolation and outside pixels set to nearest pixel
%            3: cubic interpolation and outside pixels set to zero
function imOut = imwarpfast(im, A, mode, imOutSize)


mpath = mfilename('fullpath');
mpath = fileparts(mpath)
addpath([mpath '/nonrigid_version23/functions_affine']);

%%
%generate bounds if not requested
if( isempty( imOutSize ) )
    imOutSize = max(A(1:3,1:3) * size(im)', size(im)');
end

%%
%resize input image
imSize = size(im);
if( sum(imOutSize<imSize) > 0 )
   error 'code imwarpfast is not ready to generate smaller output images than the input'
end

%im = padarray(im, max((imOutSize-imSize)/2 + 1,1), 0, 'pre');
%im = padarray(im, max((imOutSize-imSize)/2 - 1,1), 0, 'post');
im = padarray(im, imOutSize-imSize, 0, 'post');

%%
%call main routine
% Disable warning
warning('off', 'MATLAB:maxNumCompThreads:Deprecated')

%transform matrix to adapt to code convention
F = eye(4);
F(1:2,1:2) = [0 1;1 0];%flip xy coordinates

Af = A;
%Af = inv(Af);
Af(1:3,4) = Af(4,[2 1 3]);
Af(4,1:3) = 0;


Af = (F*Af) \ F;

A

Af

%we need ot center image
qq = size(im)
B = eye(4);
B(1:3,4) =  qq/ 2 + 1;

C = eye(4);
C(1:3,4) = -B(1:3,4);

Af = C * Af * B;

%call function
imOut = affine_transform(im,Af,mode);
%%
rmpath([mpath '/nonrigid_version23/functions_affine']);