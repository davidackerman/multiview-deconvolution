function [A, im] = coarseRegistrationBasedOnMicGeometry(im,angleDeg, anisotropyZ)

%save transformations
switch(angleDeg)
    
    case 0
        A = diag([1 1 anisotropyZ]);        
    case 90
        
        A = [0 0 -anisotropyZ; 0 1 0; 1 0 0];
        
    case 180
        
        A = diag([-1 1 anisotropyZ]);
        
    case 270
        
        A = [0 0 -anisotropyZ; 0 1 0; -1 0 0];
    otherwise
        error 'Angle not coded here'
        
end

%%
if( nargout > 1)
    
    %%
    %apply linear interpolation in z planes
    if( anisotropyZ > 1.0 )
        imOrig = im;
        im = zeros(floor([size(im,1) size(im,2), size(im,3)-1] .* [1 1 anisotropyZ]),'single');
        
        for ii = 1:size(im,3)
            ww = (ii-1) / anisotropyZ;
            zz = floor(ww);
            ww = ww - zz;
            im(:,:,ii) = ww .* imOrig(:,:,zz+2) + (1.0-ww) .* imOrig(:,:,zz+1);
        end
    else
        im = single(im);
    end
    
    %%
    %apply rotations
    switch(angleDeg)
        
        case 0
            %nothing to do
        case 90
            im = permute(im, [3 2 1]);
            im = flipdim(im,1);
            
        case 180
            im = flipdim(im,1);
            
        case 270
            im = permute(im, [3 2 1]);
            im = flipdim(im,1);
            im = flipdim(im,3);
            
        otherwise
            error 'Angle not coded here'
            
    end
    
end
