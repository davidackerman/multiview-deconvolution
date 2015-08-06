function xyzs = detectInterestPoints_localMaxima(im, sigma, maxNumPeaks, thrPeak, mask, verbose, debugFolder, view, minDIstanceBetweenPeaks)


imOrig = im; 
im = imdilate(im,ones(sigma,sigma,sigma));

if( isempty(debugFolder) == false )
    writeKLBstack(single(im), [debugFolder filesep 'multiview_fine_reg_DoG_view' num2str(view-1,'%.2d') '.klb']);
    if( ~isempty(mask) )
        writeKLBstack(uint8(mask), [debugFolder filesep 'multiview_fine_reg_DoGmask_view' num2str(view-1,'%.2d') '.klb']);
    end
end

im( im < thrPeak ) = 0;

if( ~isempty(mask) )
    im( mask == 0 ) = 0;
end


%detect local maxima
pp = find(imOrig == im & im >0);

[x,y,z] = ind2sub(size(im),pp);
peakVal = imOrig(pp);

XYZP = [x, y, z, peakVal];
[~,aux] = sort(peakVal,'descend');
XYZP = XYZP(aux,:);

xyzs = zeros(maxNumPeaks,4);
N = 1;
count = 2;
xyzs(1,:) = XYZP(1,:);
while(N < maxNumPeaks && count <= length(peakVal) )
   
    %check if next maxima is far away enough
    if( min(sqrt(sum((bsxfun(@minus, xyzs(1:N,1:3), XYZP(count,1:3))).^2,2))) > minDIstanceBetweenPeaks )
        N = N + 1;
        xyzs(N,:) = XYZP(count,:);
    end    
   count = count + 1;
end


if( N < maxNumPeaks )
   xyzs(N + 1:end,:) = []; 
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