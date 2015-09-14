function writeTifStack(stack,filename, compression_)


if( ndims(stack) == 2)
    stack = permute(stack,[2 1]);
elseif( ndims(stack) == 3)
    stack = permute(stack,[2 1 3]);
elseif( ndims(stack) == 4)
    stack = permute(stack,[2 1 3 4]);
end

if( nargin < 3 )
    compression_ = 'none';
end

if(ndims(stack)==3)
    imwrite(stack(:,:,1),[filename '.tif'],'Compression',compression_,'writemode','overwrite');
    
    for kk=2:size(stack,3)
        imwrite(stack(:,:,kk),[filename '.tif'],'Compression',compression_,'writemode','append');
    end
    
else%RGB stack
    
    imwrite(squeeze(stack(:,:,1,:)),[filename '.tif'],'Compression',compression_,'writemode','overwrite');
    
    for kk=2:size(stack,3)
        imwrite(squeeze(stack(:,:,kk,:)),[filename '.tif'],'Compression',compression_,'writemode','append');
    end
    
end