n_trials = 100 ;
for i_trial = 1:n_trials ,
    % Generate random spacing for the PSF
    spacing = unifrnd(0.5, 2, 1, 3) ;  % xyz
    
    % Generate random FWHM for the PSF
    FWHM = unifrnd(2*spacing, 4*spacing, 1, 3) ;
    
    % Generate the raw PSF
    raw_PSF = single(generate_PSF(spacing, FWHM)) ;
    
    % Make a random transform for each view    
    T_cond = inf ;
    while T_cond>10 ,  % Don't want condition number to be too large
        T_in_row_form = unifrnd(-2, +2, 4, 4) ;
        T_in_row_form(1:3,4) = 0 ;
        T_in_row_form(4,4) = 1 ;
        %T_in_row_form = eye(4) ;
        T_cond = cond(T_in_row_form) ;
    end        
    
    % do cpp 
    [trimmed_transformed_PSF_cpp, transformed_PSF_cpp] = transform_and_trim_PSF_cpp(T_in_row_form, raw_PSF) ;    
    
    % do Matlab
    [trimmed_transformed_PSF_matlab, transformed_PSF_matlab] = transform_and_trim_PSF_matlab(T_in_row_form, raw_PSF) ;
    
    % Compare them
    ice = is_close_enough(transformed_PSF_cpp, transformed_PSF_matlab) ;
    if ice ,
        fprintf('Close enough.\n') ;
    else
        error('Too-large deviation') ;
    end
end
fprintf('Completed %d fuzzing trials successfully.\n', n_trials) ;
