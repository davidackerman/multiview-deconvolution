function [T, im]= imRegistrationTranslationFFT(imRef, im)


options.GPU = false;
options.Power2Flag = false;%memory consumption can be ridiculous
%im can be considered as the template that is beiong moved around
fv = convnfft(im,imRef,'same',[1:max(ndims(im),ndims(imRef))],options);%fv is the same size as im

[~,pos] = max(fv(:));

[x,y,z] = ind2sub(size(fv), pos);

T = -[x,y,z] + size(imRef) / 2 - 1;

%modify im to apply transformation
if( nargout > 1)
    %im = imtranslate(im,T,'FillValues',0, 'Nearest', 'Same');
    im = imtranslate(im, T, 0, 'Nearest', 1);
end



