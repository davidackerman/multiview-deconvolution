% protocol:
% 
% 1.-Open in Fiji all the 4 views with name imWarp_Matlab_CM0?.klb resulting from coarse alignment
% 2.-Select a common landmark in all four views. xyz is 4x3 array containing the xyz position of that landmark for each view (in Fiji coordinates)
% 3.-Run this function to generate a better coarse alignment

% pathfolder = 'T:\temp\deconvolution\15_06_11_fly_functionalImage_TM000001'
function correctCoarseAlignmentUsingManualLandmarks(pathFolder, xyz)

numViews = 4;
imgBasename = [pathFolder filesep 'imWarp_Matlab_CM0'];

pathOut = [pathFolder '_manulCorr'];
if( exist(pathOut) == 0 )
    mkdir(pathOut);
end

%view 1 is the reference
for ii = 2:numViews
   %read image 
   im = readKLBstack([imgBasename num2str(ii-1) '.klb']);
   
   %calculate translation
   A = eye(4);
   T = xyz(1,:) - xyz(ii,:);
   A(4,1:3) = T;
   
   %apply tranformation
   addpath './imWarpFast/'
   im = imwarpfast(im, A, 2, size(im));
   rmpath './imWarpFast/'
   
   %write out file
   writeKLBstack(im, [pathFolder filesep 'imWarp_Matlab_CM0' num2str(ii-1) '.klb']);
   
   disp '========TODO: update tform.mat file============'
end



