function output_stack = run_matlab_imwarp_flexible(input_stack, input_origin, input_spacing, T_in_row_form, output_origin, output_spacing, output_dims)
    % Make the input frame
    input_stack_dims = size(input_stack) ;
    input_frame = frame_from_origin_spacing_and_dims(input_origin, input_spacing, input_stack_dims) ;
    
    % Make the output frame
    output_frame = frame_from_origin_spacing_and_dims(output_origin, output_spacing, output_dims) ;
                
    % Do the transform
    [output_stack, output_frame_check] = ...
        imwarp(input_stack, input_frame, affine3d(T_in_row_form), 'Interp', 'cubic', 'OutputView', output_frame) ;  %#ok<ASGLU>
end
