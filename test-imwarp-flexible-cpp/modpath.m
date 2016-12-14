function modpath()
    absolute_path_to_this_file = mfilename('fullpath') ;
    absolute_path_to_this_dir = fileparts(absolute_path_to_this_file) ;
    absolute_path_to_mvd_repo = fileparts(absolute_path_to_this_dir) ;
    absolute_path_to_klb_matlab_wrappers = fullfile(absolute_path_to_mvd_repo, 'CUDA/build/src/KLB/mex/Debug') ;
    addpath(absolute_path_to_klb_matlab_wrappers) ;
end
