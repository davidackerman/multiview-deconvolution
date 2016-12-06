function normalized_stack = run_matlab_trim_psf(input_stack)
    threshold = single(1e-10) ;
    trimmed_stack = trim_PSF_better(input_stack, threshold) ;
    trimmed_stack(trimmed_stack<0) = 0 ;  % zero out neg vals
    S = sum(sum(sum(trimmed_stack))) ;
    normalized_stack = trimmed_stack/S ;
end
