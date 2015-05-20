function xyzs = detectInterestPoints_DOG(im, sigma, maxNumPeaks, thrPeak, mask, verbose)

%TODO: better wayt to delete around local maxima. 
%TODO: multi-scale search

if( verbose > 0 )
   imOrig = im; 
end

im = single(im);
im = imgaussian(im, sigma) - imgaussian(im, sigma * 1.6);

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


%display detections
if( verbose > 0 )
    
   z = unique(xyzs(:,3));   
   for ii = 1:length(z)
       figure;
       imagesc(imOrig(:,:,z(ii)));
       colormap gray;
       pp = find( xyzs(:,3) == z(ii) );
       hold on;
       plot(xyzs(pp,2), xyzs(pp,1),'go');
       hold off;
       title(['Slice ' num2str(z(ii))]);
   end
   
   figure;plot3(xyzs(:,1),xyzs(:,2),xyzs(:,3),'o');grid on;
end