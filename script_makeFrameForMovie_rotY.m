% % example on how to call function: rotating 20 degrees between views
% % 
% % script_makeFrameForMovie_rotY('S:\temp\deconvolution\15_05_12_fly_functionalImage_cluster\multiGPUdeconv\simview3_TM????_test_mv_deconv_LR_multiGPU_iter40.klb', 2620, -2620 * 20, 20);

function script_makeFrameForMovie_rotY(imgFilePattern, TM, iniTheta, deltaTheta)

imgFilename = recoverFilenameFromPattern(imgFilePattern,TM);
im = readKLBstack(imgFilename);

%
%calculate rotation angle
theta = iniTheta + TM * deltaTheta;

%%
%define rotation matrix
%auxiliary definitions to rotate around the center
imSize = size(im);
B = eye(4);
B(4,1:3) =  imSize([2 1 3]) /2 + 1;

C = eye(4);
C(4,1:3) = -B(4,1:3);


%rotation around Y axis by 90 degrees
A = [1         0           0           0;...
     0          cosd(theta)  -sind(theta)     0;...
     0          sind(theta)  cosd(theta)  0;...
     0              0           0            1];
 
 
A = C * A * B;
 
%%
%apply rotation
addpath './imWarpFast/'
im = imwarpfast(im, A, 2, size(im));
rmpath './imWarpFast/'


%
%maximum intensity projection
imMIP = max(im, [],3);

%%
%write final image
imFilenameOut = [imgFilename(1:end-4) '_MIP.tif'];
imwrite(uint16(imMIP)', imFilenameOut);