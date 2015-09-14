function script_straightenUpDrosophila_15_05_12(TM)

%%
%parameters
imFilePattern = 'S:\temp\deconvolution\15_05_12_fly_functionalImage_cluster\multiGPUdeconv\simview3_TM????_test_mv_deconv_LR_multiGPU_iter40.raw';

%scaling
sc = 10;

%matches FIji 
XYrot = 173.5;
YZrot = -6;

%%
%read filename
imFilename = recoverFilenameFromPattern(imFilePattern, TM);
im = readRawStack(imFilename);

%%
%auxiliary definitions to rotate around the center
imSize = size(im);
B = eye(4);
B(4,1:3) =  imSize([2 1 3]) /2 + 1;

C = eye(4);
C(4,1:3) = -B(4,1:3);

%generate affine transformation for XY rotation
A1 = [cosd(XYrot) -sind(XYrot)       0     0;...
     sind(XYrot) cosd(XYrot)       0     0;...
     0              0               1     0;...
     0              0               0     1];

A1 = C * A1 * B;

%rotation around Y axis by 90 degrees
A2 = [1         0           0           0;...
     0          cosd(90)  -sind(90)     0;...
     0          sind(90)  cosd(90)  0;...
     0              0           0            1];

 
A2 = C * A2 * B;
%generate rotation around YZ 
A3 = [cosd(YZrot) -sind(YZrot)       0     0;...
     sind(YZrot) cosd(YZrot)       0     0;...
     0              0               1     0;...
     0              0               0     1];

A3 = C * A3 * B;


%composition
Afinal = A1 * A2 * A3;


%%
addpath './imWarpFast/'
im = imwarpfast(im, Afinal, 2, size(im));
rmpath './imWarpFast/'



%%
%write final image
im = uint16(sc * im);
imFilenameOut = [imFilename(1:end-4) '.klb'];
writeKLBstack(im, imFilenameOut);

