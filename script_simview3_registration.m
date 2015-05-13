
imPath = 'S:\SiMView3\15-04-24\Dme_L1_57C10-GCaMP6s_20150424_142342.corrected\SPM00\TM001445\'
imFilenameCell = {'SPM00_TM001445_CM00_CHN01.klb', 'SPM00_TM001445_CM02_CHN00.klb', 'SPM00_TM001445_CM01_CHN01.klb', 'SPM00_TM001445_CM03_CHN00.klb'};%0,90,180,270 degrees
anisotropyZ = 5.2 / 0.406;

outputFolder = 'E:\simview3_deconvolution\15_04_24\TM1445\'

%%
%fixed parameters
angles = [0 90 180 270];


%%
%prepare image reference (0 degrees angles)
filename = [imPath imFilenameCell{1}];
imRef = readKLBstack(filename);
[~,imRef] = coarseRegistrationBasedOnMicGeometry(imRef,0, anisotropyZ);

%save image reference
writeKLBstack(imRef, [outputFolder 'imRegister_Matlab_CM' num2str(0,'%.2d') '.klb']);


%genarate transformation
tformCell = cell(length(angles),1);
tformCell{1} = eye(4);

%%
%calculate alignment for each view

for ii = 2:length(angles)
    %apply coarse transformation
    filename = [imPath imFilenameCell{ii}];
    im = readKLBstack(filename);
    
    %flip and permutation
    [A, im] = coarseRegistrationBasedOnMicGeometry(im,angles(ii), anisotropyZ);
        
    %find translation    
    [T, im] = imRegistrationTranslationFFT(imRef, im);
    
    %generate affine matrix
    tformCell{ii} = [A zeros(3,1);T 1];
    
    %save image reference
    writeKLBstack(im, [outputFolder 'imRegister_Matlab_CM' num2str(ii-1,'%.2d') '.klb']);
    
end



%%
%fine registration

%use matlab stuff
disp '=====================TODO: maybe calculate contrast weights so we can weight which areas we register;================'
disp '=====================TODO: aligning all to view angle 0 is not optimal, since some views are totally complementary (so it is hard to find common points)================'