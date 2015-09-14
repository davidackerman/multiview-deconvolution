%Downsample image by 2^numLevels using a Gaussian smoothing forantialiasing
function I = stackBinning(I, numLevels)


H = [0.5 0.5];

for kk=1:numLevels
    
    bin = true;
    %average neighboring voxels
    if(ndims(I)==1)
        I=imfilter(I,H, 'same' ,'replicate');
    elseif(ndims(I)==2)
        Hx=reshape(H,[length(H) 1]);
        Hy=reshape(H,[1 length(H)]);
        I=imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
    elseif(ndims(I)==3)
        if(size(I,3)<4) % Detect if 3D or color image
            Hx=reshape(H,[length(H) 1]);
            Hy=reshape(H,[1 length(H)]);
            for k=1:size(I,3)
                I(:,:,k)=imfilter(imfilter(I(:,:,k),Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
            end
        else
            bin = false;
            %much faster mex file
            I = bin3Darray(I);
        end
    else
        error('imgaussian:input','unsupported input dimension');
    end
    
    
    %downsample XYZ
    if( bin )
        I = I(1:2:end,1:2:end,1:2:end);
    end
end