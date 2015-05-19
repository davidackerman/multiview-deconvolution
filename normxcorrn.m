function fv = normxcorrn(template, A)

if( sum( size(A) < size(template) ) > 0 )
    error 'Template needs to be smaller than A'
end

%normalize template
template = double(template);
template = template - mean(template(:));
template = template / std(template(:));

tSize = numel(template);

%flip template
for ii = 1:ndims(template)
    template = flipdim(template,ii);
end

%normalize image
Amean = double(A);
Astd = Amean.^2;

%perform convolution to calculate mean and standard values per block
h = ones(size(template,1),1);
Amean = imfilter(Amean, h, 'symmetric', 'same', 'conv');
Astd = imfilter(Astd, h, 'symmetric', 'same', 'conv');
for ii = 2:ndims(A)
    h = ones([ones(1,ii-1) size(template,ii)]) / tSize;
    
    Amean = imfilter(Amean, h, 'symmetric', 'same', 'conv');
    Astd = imfilter(Astd, h, 'symmetric', 'same', 'conv');
end

%calculate std
Astd = sqrt(Astd - (Amean.^2)) * (tSize-0.5);%-0.5 is to compensate between N and N-1 for std and get the same result as normxcorr2

%calculate correlation
options.GPU = false;
options.Power2Flag = false;%memory consumption can be ridiculous

fv = convnfft(A, template,'same',[1:max(ndims(A),ndims(template))],options);%fv is the same size as A

%normalize correlation
fv = fv ./ Astd;


