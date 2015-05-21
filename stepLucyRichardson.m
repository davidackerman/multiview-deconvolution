function imDeblurredNext = stepLucyRichardson(imOrig,psf, psfCompound, imDeblurredOld)

%{
dd = imOrig ./ (convn(imDeblurredOld,psf,'same'));
for ii = 1:ndims(psf)
   psf = flip(psf,ii); 
end
imDeblurredNext = convn( dd , psf, 'same');
%}


options.GPU = false;
options.Power2Flag = true;

dd = imOrig ./ (convnfft(imDeblurredOld,psf,'same',1:ndims(psf),options));

dd( isnan(dd) ) = 1;%as if this pixel was perfect already

for ii = 1:ndims(psf)
   psf = flip(psf,ii); 
end

if( isempty(psfCompound) )
    imDeblurredNext = convnfft( dd , psf, 'same',1:ndims(psf),options);
else
    imDeblurredNext = convnfft( dd , psf .* psfCompound, 'same',1:ndims(psf),options);
end

