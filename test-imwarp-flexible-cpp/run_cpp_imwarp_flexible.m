function output_stack = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, T_in_row_form, output_origin, output_spacing, output_dims)
    writeKLBstack(input_stack, 'input_stack.klb') ;
    write_as_raw('input_origin.raw', input_origin) ;
    write_as_raw('input_spacing.raw', input_spacing) ;
    write_as_raw('T_in_row_form.raw', T_in_row_form) ;
    write_as_raw('output_origin.raw', output_origin) ;
    write_as_raw('output_spacing.raw', output_spacing) ;
    % Write output dims file
    write_as_raw('output_dims.raw', int64(output_dims)) ;
    
    % Get the executable name
    absolute_path_to_this_file = mfilename('fullpath') ;
    absolute_path_to_repo_root = fileparts(fileparts(absolute_path_to_this_file)) ;
    absolute_path_to_executable = fullfile(absolute_path_to_repo_root, 'CUDA/build/src/imwarp_flexible/Debug/run_imwarp_flexible.exe' )
    
    command_line = ...
        sprintf('%s %s %s %s %s %s %s %s %s', ...
                absolute_path_to_executable, ...
                'input_stack.klb', ...
                'input_origin.raw', ...
                'input_spacing.raw', ...
                'T_in_row_form.raw', ...
                'output_origin.raw', ...
                'output_spacing.raw', ...
                'output_dims.raw', ...
                'output.klb') ;
    fprintf('%s\n', command_line) ;        
    status = system(command_line) ;

    if status~=0 ,
        error('run_imwarp_flexible.exe failed with status %d', status) ;
    end

    output_stack = readKLBstack('output.klb') ;    
end
