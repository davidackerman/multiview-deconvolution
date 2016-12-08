n_trials = 100 ;
for i_trial = 1:n_trials ,
    % Fixed things
    input_origin = [0.5 0.5 0.5] ;
    output_spacing = ones(1,3) ;
    
    % Generate nonrandom spacing for the PSF
    input_spacing = ones(1,3) ;
    
    % Generate random FWHM for the PSF
    FWHM = unifrnd(2*input_spacing, 4*input_spacing, 1, 3) ;
    
    % Generate the raw PSF
    input_stack = generate_PSF(input_spacing, FWHM) ;
    input_dims = size(input_stack) ;
    
    % Generate a random xform matrix for this view
    T_cond = inf ;
    while T_cond>10 ,  % Don't want condition number to be too large
        T_in_row_form = unifrnd(-2, +2, 4, 4) ;
        T_in_row_form(1:3,4) = 0 ;
        T_in_row_form(4,4) = 1 ;
        T_cond = cond(T_in_row_form) ;
    end    

    % Run the cpp code
    [output_dims_cpp, output_origin_cpp] = ...
        run_transform_lattice_cpp(T_in_row_form, ...
                                  input_dims, input_origin, input_spacing, ...
                                  output_spacing) ;
    [output_dims_matlab, output_origin_matlab] = ...
        run_transform_lattice_matlab(T_in_row_form, ...
                                     input_dims, input_origin, input_spacing, ...
                                     output_spacing) ;
                                                             
    output_dims_cpp
    output_dims_matlab
    output_origin_cpp
    output_origin_matlab
    
    % Compare them
    if ~isequal(output_dims_cpp, output_dims_matlab) ,
        error('Output dims are not the same') ;
    end
    if max(abs(output_origin_cpp-output_origin_matlab))>1e-12 ,
        error('Output origins differ by too much') ;
    end    
end
fprintf('Passed %d fuzzing trials!\n', n_trials) ;
