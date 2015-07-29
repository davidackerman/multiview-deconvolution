%  mode: If 0: linear interpolation and outside pixels set to nearest pixel
%            1: linear interpolation and outside pixels set to zero
%            2: cubic interpolation and outside pixels set to nearest pixel
%            3: cubic interpolation and outside pixels set to zero
function imOut = imwarpfast(im, A, mode, imOutSize)


mpath = mfilename('fullpath');
mpath = fileparts(mpath);
addpath([mpath '/nonrigid_version23/functions_affine']);

%%
%matrix flip xy coordinates
F = eye(4);
F(1:2,1:2) = [0 1;1 0];

%%
%generate bounds if not requested
D = eye(4);%displacement in case output box lower bound is not [0 0 0]
if( isempty( imOutSize ) )    
    error 'imwarpfast: code not ready to emulate imwarp when no bnounding box is giveb'
    ROI = round(findBoundingBox(F*A', size(im))); %works with PSF example
    %ROI = round(findBoundingBox(A', size(im)))%works for test_A
    imOutSize = diff(ROI);
    D(1:3,4) = ROI(1,[2 1 3]) / 2 + 1; %works for roation on XY with C * Af * B * D;
    %D(1:3,4) = ROI(1,[2 1 3]); %works for pure translation with C * Af * B * D; (D can go anywhere here)
end

%%
%resize input image
imSize = size(im);
if( sum(imOutSize < imSize) > 0 )
   error 'code imwarpfast is not ready to generate smaller output images than the input'
end

%im = padarray(im, max((imOutSize-imSize)/2 + 1,1), 0, 'pre');
%im = padarray(im, max((imOutSize-imSize)/2 - 1,1), 0, 'post');
im = padarray(im, imOutSize-imSize, 0, 'post');

%%
%call main routine
% Disable warning
warning('off', 'MATLAB:maxNumCompThreads:Deprecated')

Af = (A'*F)\F;


%we need to center image
B = eye(4);
B(1:3,4) =  size(im) / 2 + 1;

C = eye(4);
C(1:3,4) = -B(1:3,4);

Af = C * Af * B * D;

%call function
imOut = affine_transform(im,Af,mode);
%%
rmpath([mpath '/nonrigid_version23/functions_affine']);