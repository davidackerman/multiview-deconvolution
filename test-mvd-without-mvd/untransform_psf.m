function raw_PSF = untransform_psf(desired_trimmed_transformed_PSF, T_in_row_form)
    % Generate the raw PSF
    [raw_PSF, spacing, FWHM] = generate_PSF() ; %#ok<ASGLU>

    % Generate a reference frame for the raw PSF, which is just the default
    % reference frame that imwarp() assumes if you don't pass one in.
    untransformed_PSF_frame = ...
        imref3d(size(raw_PSF), ...
                0.5+[0 size(raw_PSF,2)], ...
                0.5+[0 size(raw_PSF,1)], ...
                0.5+[0 size(raw_PSF,3)]) ;

    % Do the transform in the default way        
    [transformed_PSF, untrimmed_PSF_frame] = ...
        imwarp(raw_PSF, untransformed_PSF_frame, affine3d(T_in_row_form), 'Interp', 'cubic') ;
    %default_target_PSF_size = size(default_target_PSF)
    %default_target_frame

    % Trim the target PSF
    [trimmed_transformed_PSF, trimmed_transformed_frame] = ...
        trim_and_normalize_PSF_with_reference_frame(transformed_PSF, untrimmed_PSF_frame) ;  %#ok<ASGLU>
    %trimmed_default_target_PSF_size = size(trimmed_transformed_PSF)

    % Check that it equals the desired PSF
    did_get_right_answer = ...
        isequal(trimmed_transformed_PSF, desired_trimmed_transformed_PSF) ;  % this is true, but the reference frames are all wrong
    if ~did_get_right_answer ,
        error('Unable to determine the raw PSF to match given ideal trimmed transformed PSF') ;
    end
end
