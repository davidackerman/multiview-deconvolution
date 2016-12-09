n_views = 3 ;
n_trials = 100 ;
for i_trial = 1:n_trials ,
    % Generate random spacing for the PSF
    spacing = unifrnd(0.5, 2, 1, 3) ;  % xyz
    
    % Generate random FWHM for the PSF
    FWHM = unifrnd(2*spacing, 4*spacing, 1, 3) ;
    
    % Generate the raw PSF
    raw_PSF = single(generate_PSF(spacing, FWHM)) ;
    
    % Make a random transform for each view    
    Ts_in_row_form = cell(1, n_views) ;
    for i_view = 1:n_views ,
        T_cond = inf ;
        while T_cond>10 ,  % Don't want condition number to be too large
            T_in_row_form = unifrnd(-2, +2, 4, 4) ;
            T_in_row_form(1:3,4) = 0 ;
            T_in_row_form(4,4) = 1 ;
            %T_in_row_form = eye(4) ;
            T_cond = cond(T_in_row_form) ;
        end        
        Ts_in_row_form{i_view} = T_in_row_form ;
    end
    
    % do cpp 
    [trimmed_transformed_PSFs_cpp, transformed_PSFs_cpp] = transform_and_trim_PSFs_cpp(Ts_in_row_form, raw_PSF) ;    
    
    % do Matlab
    [trimmed_transformed_PSFs_matlab, transformed_PSFs_matlab] = transform_and_trim_PSFs_matlab(Ts_in_row_form, raw_PSF) ;
    
    % Compare them
    for i_view = 1:n_views ,
        trimmed_transformed_PSF_cpp = trimmed_transformed_PSFs_cpp{i_view} ;
        transformed_PSF_cpp = transformed_PSFs_cpp{i_view} ;
        trimmed_transformed_PSF_matlab = trimmed_transformed_PSFs_matlab{i_view} ;
        transformed_PSF_matlab = transformed_PSFs_matlab{i_view} ;
        ice = is_close_enough(trimmed_transformed_PSF_cpp, trimmed_transformed_PSF_matlab) ;
        if ice ,
            fprintf('Close enough.\n') ;
        else
            error('Too-large deviation') ;
        end
    end
end
fprintf('Completed %d fuzzing trials successfully.\n', n_trials) ;
