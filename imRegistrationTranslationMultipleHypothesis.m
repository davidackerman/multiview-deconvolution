% we use cross correlation and obtain multiple hypothesis for the
% translation

%T us the translation that we need to apply to im to register with imRef
function T = imRegistrationTranslationMultipleHypothesis(imRef, im, numHypothesis)

%TODO: local maxima search (window is too small: it has ot be adaptive)

winErase = 10;

options.GPU = false;
options.Power2Flag = false;%memory consumption can be ridiculous
%im can be considered as the template that is beiong moved around

fv = convnfft(imRef, im,'same',[1:max(ndims(im),ndims(imRef))],options);%fv is the same size as imRef

T = zeros(numHypothesis, 4);
for ii = 1:numHypothesis

    [val,pos] = max(fv(:));
    [x,y,z] = ind2sub(size(fv), pos);
    T(ii,:) = [-[x,y,z] + size(im) / 2 - 1, val];
    
    fv(max(x-winErase,1):min(x+winErase, size(fv,1)), max(y-winErase,1):min(y+winErase, size(fv,2)), max(z-winErase,1):min(z+winErase, size(fv,3))) = -1;
end





