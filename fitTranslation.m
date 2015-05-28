function [A, nccVal] = fitTranslation(imRef, im, winRadius)


%find a block around brightest point
[val, pos] = max(im(:));

[x,y,z] = ind2sub(size(im),pos);

x = max(winRadius+1,x);
y = max(winRadius+1,y);
z = max(winRadius+1,z);
x = min(size(im,1)-winRadius-1,x);
y = min(size(im,2)-winRadius-1,y);
z = min(size(im,3)-winRadius-1,z);

imBlock = im(x-winRadius+1:x+winRadius,y-winRadius+1:y+winRadius,z-winRadius+1:z+winRadius);

%find translation
Taux = imRegistrationTranslationMultipleHypothesis(imRef, imBlock, 1, 0.7);%it has to be a good match

if( isempty(Taux) )
    Taux = zeros(1,4);
else
    offset = [x,y,z] - size(imRef)/2;
    Taux(1:3) = Taux(1:3) + offset;
end


A = eye(4);
A(4,1:3) = -Taux([2 1 3]);
nccVal = Taux(4);