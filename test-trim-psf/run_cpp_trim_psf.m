function output_stack = run_cpp_trim_psf(input_stack)
    writeKLBstack(input_stack, 'input_stack.klb') ;
    
    % Get the executable name
    absolute_path_to_this_file = mfilename('fullpath') ;
    absolute_path_to_repo_root = fileparts(fileparts(absolute_path_to_this_file)) ;
    absolute_path_to_executable = fullfile(absolute_path_to_repo_root, 'CUDA/build/src/Debug/run_trim_psf.exe' )
    
    command_line = ...
        sprintf('%s %s %s', ...
                absolute_path_to_executable, ...
                'input_stack.klb', ...
                'output_stack.klb') ;
    fprintf('%s\n', command_line) ;        
    status = system(command_line) ;

    if status~=0 ,
        error('run_trim_psf.exe failed with status %d', status) ;
    end

    output_stack = readKLBstack('output_stack.klb') ;    
end
