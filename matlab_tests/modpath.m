function old_path = modpath(release_or_debug)
    if exist('release_or_debug', 'var') && isequal(release_or_debug,'release') ,
        release_or_debug_dir_name = 'Release' ;
    else
        release_or_debug_dir_name = 'Debug' ;
    end

    absolute_path_to_this_file = mfilename('fullpath') ;
    absolute_path_to_this_dir = fileparts(absolute_path_to_this_file) ;
    absolute_path_to_mvd_repo = fileparts(absolute_path_to_this_dir) ;
    absolute_path_to_klb_matlab_wrappers = fullfile(absolute_path_to_mvd_repo, 'CUDA', 'build', 'src', 'klb', 'mex', release_or_debug_dir_name) ;
    old_path = addpath(absolute_path_to_klb_matlab_wrappers) ;
end
