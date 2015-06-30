function [im, A] = applyRegistrationFromFiji(im , imRefSize, tformFiji, ROI)


%%
%other constants
F = eye(4); %to flip XY coordinates
F(1:2,1:2) = [0 1;1 0];

%calculate transformation
A = eye(1);
for jj = 1:length(tformFiji)
    A = tformFiji{jj} * A;
end
A = F \ (A*F);

%apply tranformation
addpath './imWarpFast/'
im = single(imwarpfast(im, A, 2, imRefSize));
rmpath './imWarpFast/'

%crop output
if ( isempty(ROI) == false )
   im = im(ROI(1,1):ROI(2,1), ROI(1,2):ROI(2,2), ROI(1,3):ROI(2,3));
end



