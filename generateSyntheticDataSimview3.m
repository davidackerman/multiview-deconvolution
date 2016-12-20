%function generateSyntheticDataSimview3()

%%
%parameters
imSize = [128 45 35];

muNoise = 100.1;
sigmaNoise = 3.2;

%object location
muObj = [65   20    12;...
         20   31    20;...
         100  23    22];
sigmaObj = [2   4   7] ;

photonsObj = [100  1000   10000];

%%
%generate image
imOrig = zeros(imSize);

for ii = 1:length(sigmaObj)    
   cc = muObj(ii,:);   
   imOrig(cc(1)-sigmaObj(ii):cc(1)+sigmaObj(ii),cc(2)-sigmaObj(ii):cc(2)+sigmaObj(ii), cc(3)-sigmaObj(ii):cc(3)+sigmaObj(ii)) = photonsObj(ii);   
end

%save ground truth
