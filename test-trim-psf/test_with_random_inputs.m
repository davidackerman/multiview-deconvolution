n_trials = 100 ;
for i_trial = 1:n_trials ,
    % Generate random spacing for the PSF
    input_spacing = unifrnd(0.5, 2, [1 3]) ;  % xyz
    
    % Generate random FWHM for the PSF
    FWHM = unifrnd(2*input_spacing, 4*input_spacing, 1, 3) ;
    
    % Generate the raw PSF
    input_stack = generate_PSF(input_spacing, FWHM) ;
    
    % Generate a random xform matrix for this view
    T_cond = inf ;
    while T_cond>10 ,  % Don't want condition number to be too large
        T_in_row_form = unifrnd(-2, +2, 4, 4) ;
        T_in_row_form(1:3,4) = 0 ;
        T_in_row_form(4,4) = 1 ;
        T_cond = cond(T_in_row_form) ;
    end
    
    % Generate an input origin
    input_origin = normrnd(0.5, 1, [1 3]) ;

    % Make an input frame
    input_stack_dims = size(input_stack) ;
    input_frame = frame_from_origin_spacing_and_dims(input_origin, input_spacing, input_stack_dims) ;
    
    % Call imwarp just to calculate the default output frame
    [~, default_output_frame] = ...
        imwarp(input_stack, input_frame, affine3d(T_in_row_form)) ;
    
    % Extract the origin, spacing, dims from the frame
    [default_output_origin, default_output_spacing, default_output_dims] = origin_spacing_and_dims_from_frame(default_output_frame) ;
    
    % Fuzz the origin and spacing a bit, but leave dims alone
    output_origin = default_output_origin + normrnd(0.5, 1, [1 3]) ;
    output_spacing = default_output_spacing .* unifrnd(0.8, 1.2, [1 3]) ;
    output_dims = default_output_dims ;
    
    % Convert stuff to desired types
    input_stack_single = single(input_stack) ;
    input_origin_single = single(input_origin) ;
    input_spacing_single = single(input_spacing) ;
    T_in_row_form_single = single(T_in_row_form) ;
    output_origin_single = single(output_origin) ;
    output_spacing_single = single(output_spacing) ;
    output_dims_uint64 = uint64(output_dims) ;
    
    % Make sure all the imwarp_flexible inputs are single
    assert(isa(input_stack_single, 'single')) ;   
    assert(isa(input_origin_single, 'single')) ;   
    assert(isa(input_spacing_single, 'single')) ;   
    assert(isa(T_in_row_form_single, 'single')) ;   
    assert(isa(output_origin_single, 'single')) ;   
    assert(isa(output_spacing_single, 'single')) ;   
    assert(isa(output_dims_uint64, 'uint64')) ;   
    
    % Compute the transformed PSF in Matlab
    transformed_psf = ...
        run_matlab_imwarp_flexible(input_stack_single, input_origin_single, input_spacing_single, ...
                                   T_in_row_form_single, ...
                                   output_origin_single, output_spacing_single, output_dims_uint64) ;
        
    assert(isa(transformed_psf, 'single')) ;   

    % Trim the psf in matlab
    trimmed_psf_matlab = run_matlab_trim_psf(transformed_psf) ;
    
    % Trim the psf in C++
    trimmed_psf_cpp = run_cpp_trim_psf(transformed_psf) ;

    % Compare them
    [are_close, are_same_size] = is_close_enough(trimmed_psf_cpp, trimmed_psf_matlab) ;
    if are_close ,
        fprintf('Stacks are close enough!\n') ;
    else
        if are_same_size ,
            error('Stacks are same size, but values differ too much') ;
        else
            error('Stacks are different sizes') ;
        end
    end
end
fprintf('Passed %d fuzzing trials!\n', n_trials) ;
