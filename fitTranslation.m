function [A, nccVal] = fitTranslation(imRef, im, winRadius)


%find a block around brightest point
[val, pos] = max(im(:));

[x,y,z] = ind2sub(size(im),pos);

imBlock = im(x-winRadius+1:x+winRadius,y-winRadius+1:y+winRadius,z-winRadius+1:z+winRadius);

%find translation
Taux = imRegistrationTranslationMultipleHypothesis(imRef, imBlock, 1, 0);

if( isempty(Taux) )
    Taux = zeros(1,4);
end

offset = [x,y,z] - size(imRef)/2;
Taux(1:3) = Taux(1:3) + offset;

A = eye(4);
A(4,1:3) = -Taux([2 1 3]);
nccVal = Taux(4);