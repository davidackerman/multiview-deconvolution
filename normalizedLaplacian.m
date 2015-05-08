%In cartesian coordinates laplacian(I) = \sum_i d^2I/d^2x_i (sum of second
%derivatives)
function L = normalizedLaplacian(im, sigma)


%basic Gaussian derivative kernel
hx = ndgauss(8*sigma,sigma,'der',[1 0],'normalize',1);

%finite differences
hxf = [-1 0 1];


%calculate L1 norm of the gradient
gL1 = zeros(size(im),'single');
for ii = 1:ndims(im)
    %first derivative of the image with normalization
    switch(ii)
        case 1
            h = hx;
        case 2
            h = hx';
        case 3
            h = zeros([1 1 length(hx)]);
            h(1,1,:) = hx;
    end
    
    gL1 = gL1 + abs(imfilter(single(im), h, 'symmetric','same','conv'));
end

%calculate laplacian and normalization by absolute norm
L = zeros(size(im),'single');
for ii = 1:ndims(im)
    %first derivative of the image with normalization
    switch(ii)
        case 1
            h = hx;
            hf = hxf;
        case 2
            h = hx';
            hf = hxf';
        case 3
            h = zeros([1 1 length(hx)]);
            h(1,1,:) = hx;
            hf = zeros([1 1 length(hxf)]);
            hf(1,1,:) = hxf;
    end
    
    aux = imfilter(single(im), h, 'symmetric','same','conv') ./ (gL1 + eps('single'));
    
    %apply divergence (easy since it is just a bunch of sign)
    L = L + imfilter(aux , hf, 'symmetric','same','conv');    
end