function output_stack = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, T_in_row_form, output_origin, output_spacing, output_dims)
    writeKLBstack(input_stack, 'input_stack.klb') ;
    writeKLBstack(input_origin, 'input_origin.klb') ;
    writeKLBstack(input_spacing, 'input_spacing.klb') ;
    writeKLBstack(T_in_row_form, 'T_in_row_form.klb') ;
    writeKLBstack(output_origin, 'output_origin.klb') ;
    writeKLBstack(output_spacing, 'output_spacing.klb') ;
    % Write output dims file
    fid = fopen('output_dims.txt', 'wt') ;
    fprintf(fid, '%d  %d  %d\n', output_dims) ;
    fclose(fid) ;
    
    % Get the executable name
    absolute_path_to_this_file = mfilename('fullpath') ;
    absolute_path_to_repo_root = fileparts(fileparts(absolute_path_to_this_file)) ;
    absolute_path_to_executable = fullfile(absolute_path_to_repo_root, 'CUDA/build/src/imwarp_flexible/Debug/run_imwarp_flexible.exe' )
    
    command_line = ...
        sprintf('%s %s %s %s %s %s %s %s %s', ...
                absolute_path_to_executable, ...
                'input_stack.klb', ...
                'input_origin.klb', ...
                'input_spacing.klb', ...
                'T_in_row_form.klb', ...
                'output_origin.klb', ...
                'output_spacing.klb', ...
                'output_dims.txt', ...
                'output.klb') ;
    fprintf('%s\n', command_line) ;        
    status = system(command_line) ;

    if status~=0 ,
        error('run_imwarp_flexible.exe failed with status %d', status) ;
    end

    output_stack = readKLBstack('output.klb') ;    
end
