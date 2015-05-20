%function testTranslationRegistration()

imRef = zeros([64,32,16]);
imRef(30:34,14:18,6:10) = 100;

im = imRef(4:48,8:32,1:16);


[T, imA]= imRegistrationTranslationFFT(imRef, im);


imRefAux = imRef(1:size(imA,1),1:size(imA,2),1:size(imA,3));


T
sum(imRefAux(:) .* imA(:) ) == 125 * 100 ^2

Tgt = [4 8 1] - 1

