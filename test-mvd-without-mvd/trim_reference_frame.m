function trimmed_frame = trim_reference_frame(frame, stack_size, n_voxels_to_erase_on_either_side_plus_one) 
    % N.B.: n_voxels_to_erase_on_either_side_plus_one is 3 x 1, and is in order
    % y, x, z; *not* x, y, z
    % stack_size is the size of the stack associated with reference frame
    % frame
    if iscolumn(n_voxels_to_erase_on_either_side_plus_one) ,
        n_voxels_to_erase_on_either_side = n_voxels_to_erase_on_either_side_plus_one' - 1 ;
    else
        n_voxels_to_erase_on_either_side = n_voxels_to_erase_on_either_side_plus_one - 1 ;
    end
    
    trimmed_frame_x_world_limits = ...
        frame.XWorldLimits + frame.PixelExtentInWorldX * n_voxels_to_erase_on_either_side(2) * [+1 -1] ;
    trimmed_frame_y_world_limits = ...
        frame.YWorldLimits + frame.PixelExtentInWorldY * n_voxels_to_erase_on_either_side(1) * [+1 -1] ;
    trimmed_frame_z_world_limits = ...
        frame.ZWorldLimits + frame.PixelExtentInWorldZ * n_voxels_to_erase_on_either_side(3) * [+1 -1] ;    
    
    trimmed_stack_size = stack_size-2*n_voxels_to_erase_on_either_side ;
    trimmed_frame = ...
       imref3d(trimmed_stack_size, trimmed_frame_x_world_limits, trimmed_frame_y_world_limits, trimmed_frame_z_world_limits) ;
end
