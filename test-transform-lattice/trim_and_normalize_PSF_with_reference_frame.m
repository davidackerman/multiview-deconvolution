function [y, y_frame] = trim_and_normalize_PSF_with_reference_frame(x, x_frame)        
    % Convert to single-precision
    transformed_PSF_1 = single(x) ;
    transformed_PSF_1_reference_frame = x_frame ;
    
    % make sure it does not have negative values from cubic "ringing"
    transformed_PSF_2 = max(transformed_PSF_1,0) ;
    transformed_PSF_2_reference_frame = transformed_PSF_1_reference_frame ;
    
    % crop PSF to reduce it in size
    [transformed_PSF_3, n_voxels_to_erase_on_either_side_plus_one] = trim_PSF(transformed_PSF_2, 1e-10) ;    
    %trimmed_PSF = PSF(erase(1):end-erase(1)+1, erase(2):end-erase(2)+1, erase(3):end-erase(3)+1);
    transformed_PSF_3_reference_frame = ...
        trim_reference_frame(transformed_PSF_2_reference_frame, size(transformed_PSF_2), n_voxels_to_erase_on_either_side_plus_one) ;
    
    % normalize PSF
    y = transformed_PSF_3 / sum(transformed_PSF_3(:)) ;    
    y_frame = transformed_PSF_3_reference_frame ;
end
