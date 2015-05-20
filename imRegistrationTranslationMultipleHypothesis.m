% we use cross correlation and obtain multiple hypothesis for the
% translation

%T us the translation that we need to apply to im to register with imRef
function T = imRegistrationTranslationMultipleHypothesis(imRef, im, numHypothesis)

%TODO: local maxima search (window is too small: it has ot be adaptive with some sort of watershed)

thrNCC = 0.5;

options.GPU = false;
options.Power2Flag = false;%memory consumption can be ridiculous
%im can be considered as the template that is beiong moved around

fv = normxcorrn(im, imRef);%fv is the same size as imRef

fv( fv < thrNCC ) = 0;

fvL = bwlabeln(fv > 0, 6);

T = zeros(numHypothesis, 4);
for ii = 1:numHypothesis

    [val,pos] = max(fv(:));
    
    if( val <= thrNCC )
        break;
    end
    
    [x,y,z] = ind2sub(size(fv), pos);
    T(ii,:) = [-[x,y,z] + size(fv) / 2 , val];
    
    %delete local maxima
    fv( fvL == fvL(pos) ) = 0;
end

if( ii < numHypothesis )
   T(ii+1:end,:) = []; 
end




