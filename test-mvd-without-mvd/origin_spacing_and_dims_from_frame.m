function [origin, spacing, dims] = origin_spacing_and_dims_from_frame(frame)
    xl = frame.XWorldLimits ;
    yl = frame.YWorldLimits ;
    zl = frame.ZWorldLimits ;
    origin = [xl(1) yl(1) zl(1)] ;
    
    spacing = [frame.PixelExtentInWorldX frame.PixelExtentInWorldY frame.PixelExtentInWorldZ] ;
    
    dims = frame.ImageSize ;
end
