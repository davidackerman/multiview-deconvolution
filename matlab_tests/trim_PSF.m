function [trimmed_psf, n_voxels_to_trim_plus_one, n_voxels_to_trim] = trim_PSF_better(psf, threshold)
    % This does the same thing as trim_PSF(), but returns one extra argument, and the code is more readable.
    % Did some fuzzing style testing to confirm that this always returns the same thing as trim_PSF() for
    % the first two return values.
    if nargin < 2 ,
        threshold = 1e-8 ;
    end
    
    is_above_threshold = (psf > threshold) ;

    n_dims = ndims(psf) ;
    %if (n_dims~=3) ,
    %    error('This function only works for 3D arrays') ;
    %end
    
    %n_voxels_to_trim_plus_one = zeros(n_dims,1) ;  % on either side of the array, per dimension
    n_voxels_to_trim = zeros(n_dims,1) ;  % on either side of the array, per dimension
    for i_dim = 1:n_dims ,
        % On this iteration, we determine how many voxels to trim in
        % dimension i_dim.
        n_els_this_dim = size(psf, i_dim) ;
        
        % Reduce all the dimensions that are not of interest
        is_any_voxel_above_threshold_in_subarray = is_above_threshold ;  
        for j_dim = n_dims:-1:1 ,
            % In this iter, we collapse dimension j_dim, unless it's the
            % dimension of interest (i_dim).
            if i_dim ~= j_dim ,
                is_any_voxel_above_threshold_in_subarray = any(is_any_voxel_above_threshold_in_subarray, j_dim) ;
            end
        end
        is_any_voxel_above_threshold_in_subarray = reshape(is_any_voxel_above_threshold_in_subarray,[n_els_this_dim 1]) ;  
          % All dimensions but one (i_dim) are singleton already, but make
          % it col vector for convenience.

        % Figure out how much we can cut: the trimming has to be symmetric to
        % keep PSF centered.
        index_of_first_nonempty_subarray = find(is_any_voxel_above_threshold_in_subarray, 1, 'first') ;
        if isempty(index_of_first_nonempty_subarray) ,
            % In this case, we pretend the first element was superthreshold to prevent trimming away to nothingness
            index_of_first_subarray_to_keep = 1 ;
        else
            index_of_first_subarray_to_keep = max(1, index_of_first_nonempty_subarray - 2 ) ;  % We add a little padding around the nonempty part, if possible
        end
        index_of_last_nonempty_subarray = find(is_any_voxel_above_threshold_in_subarray, 1, 'last' ) ;
        if isempty(index_of_last_nonempty_subarray) ,
            % In this case, we pretend the last element was superthreshold to prevent trimming away to nothingness
            index_of_the_last_subarray_to_keep = n_els_this_dim ;
        else        
            index_of_the_last_subarray_to_keep = min(n_els_this_dim, index_of_last_nonempty_subarray + 2 ) ;
        end
%         n_voxels_to_trim_plus_one(i_dim) = min(index_of_first_subarray_to_keep, ...
%                                                n_els_this_dim - index_of_the_last_subarray_to_keep + 1 ) ;
        n_voxels_to_trim_at_low_end = index_of_first_subarray_to_keep-1 ;
        n_voxels_to_trim_at_high_end = n_els_this_dim - index_of_the_last_subarray_to_keep ;
        
        n_voxels_to_trim(i_dim) = min(n_voxels_to_trim_at_low_end, n_voxels_to_trim_at_high_end ) ;  % want to trim same number on each end
    end

%     result = psf(n_voxels_to_trim_plus_one(1):end-n_voxels_to_trim_plus_one(1)+1, ...
%                  n_voxels_to_trim_plus_one(2):end-n_voxels_to_trim_plus_one(2)+1, ...
%                  n_voxels_to_trim_plus_one(3):end-n_voxels_to_trim_plus_one(3)+1) ;
    if n_dims==2 ,
        trimmed_psf = psf(n_voxels_to_trim(1)+1:end-n_voxels_to_trim(1), ...
                          n_voxels_to_trim(2)+1:end-n_voxels_to_trim(2)) ;
    elseif n_dims==3 ,
        trimmed_psf = psf(n_voxels_to_trim(1)+1:end-n_voxels_to_trim(1), ...
                          n_voxels_to_trim(2)+1:end-n_voxels_to_trim(2), ...
                          n_voxels_to_trim(3)+1:end-n_voxels_to_trim(3)) ;
    else
        error('Unsupported arity') ;
    end
    n_voxels_to_trim_plus_one = n_voxels_to_trim+1 ;         
end
