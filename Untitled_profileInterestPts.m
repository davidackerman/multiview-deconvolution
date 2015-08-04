
imKK = single(im);
imKK = imgaussian(imKK, sigma) - imgaussian(imKK, sigma * 1.6);


im( im < thrPeak ) = 0;

if( ~isempty(mask) )
    im( mask == 0 ) = 0;
end

L = bwlabeln(im > 0, 6);

xyzs = zeros(maxNumPeaks,4);

for ii = 1:maxNumPeaks
   [val,pos] = max(im(:)); 
   if ( val < thrPeak )
       break;
   end
    [x,y,z] = ind2sub(size(im), pos);    
    xyzs(ii,:) = [x, y, z, val];
    
    im( L == L(pos) ) = 0;
end

if( ii < maxNumPeaks )
   xyzs(ii:end,:) = []; 
end
