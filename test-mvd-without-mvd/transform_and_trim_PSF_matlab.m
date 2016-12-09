function [trimmed_transformed_PSF, transformed_PSF] = transform_and_trim_PSF_matlab(T_in_row_form, raw_PSF)
    % Generate a reference frame for the raw PSF, which is just the default
    % reference frame that imwarp() assumes if you don't pass one in.
    untransformed_PSF_frame = ...
        imref3d(size(raw_PSF), ...
                0.5+[0 size(raw_PSF,2)], ...
                0.5+[0 size(raw_PSF,1)], ...
                0.5+[0 size(raw_PSF,3)]) ;

    % Do once just to get the output frame
    [~, transformed_PSF_frame] = ...
        imwarp(raw_PSF, untransformed_PSF_frame, affine3d(T_in_row_form), 'Interp', 'cubic') ;
    [transformed_origin, transformed_spacing, transformed_dims] = origin_spacing_and_dims_from_frame(transformed_PSF_frame)
                
    % Do the transform for real
    transformed_PSF = ...
        imwarp(raw_PSF, untransformed_PSF_frame, affine3d(T_in_row_form), 'Interp', 'cubic', 'OutputView', transformed_PSF_frame) ;
    transformed_PSF_size_matlab = size(transformed_PSF)

    % Trim the target PSF
    [trimmed_transformed_PSF, trimmed_transformed_frame] = ...
        trim_and_normalize_PSF_with_reference_frame(transformed_PSF, transformed_PSF_frame) ;  %#ok<ASGLU>
    %trimmed_default_target_PSF_size = size(trimmed_transformed_PSF)

    assert(isa(trimmed_transformed_PSF, 'single')) ;
    assert(isa(transformed_PSF, 'single')) ;    
end
