function [output_dims, output_origin] =  run_transform_lattice_matlab(T_in_row_form, ...
                                                                      input_dims, input_origin, input_spacing, ...
                                                                      output_spacing)

    % Make an input frame
    input_frame = frame_from_origin_spacing_and_dims(input_origin, input_spacing, input_dims) ;
    
    % Call imwarp just to calculate the default output frame
    [~, output_frame] = ...
        imwarp(zeros(input_dims), input_frame, affine3d(T_in_row_form)) ;
    
    % Extract the origin, spacing, dims from the frame
    [output_origin, output_spacing_check, output_dims] = origin_spacing_and_dims_from_frame(output_frame) ;
    
    % Check that the spacing is equal to the desired spacing, which would
    % be miraculous unless output_spacing is ones(1,3)
    if max(abs(output_spacing_check-output_spacing))>1e-12 ,
        error('Output spacing is not as desired') ;
    end
end
