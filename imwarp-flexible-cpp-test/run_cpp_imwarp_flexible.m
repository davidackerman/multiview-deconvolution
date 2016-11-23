function output_stack = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, A, output_origin, output_spacing, output_dims)   %#ok<INUSD,INUSL>
    writeKLBstack(input_stack, 'input.klb') ;
    
    input_origin_and_spacing = [input_origin input_spacing] ; %#ok<NASGU>
    save('input_origin_and_spacing.txt', 'input_origin_and_spacing', '-ascii') ;
    
    save('affine_transform_matrix_in_y_equals_Ax_form_row_major.txt', 'A', '-ascii') ;

    output_origin_and_spacing = [output_origin output_spacing] ; %#ok<NASGU>
    save('output_origin_and_spacing.txt', 'output_origin_and_spacing', '-ascii') ;
    
    fid = fopen('output_dimensions.txt', 'wt') ;
    fprintf(fid, '%d  %d  %d\n', output_dims) ;
    fclose(fid) ;
    
    % Get the executable name
    absolute_path_to_this_file = mfilename('fullpath') ;
    absolute_path_to_repo_root = fileparts(fileparts(absolute_path_to_this_file)) ;
    absolute_path_to_executable = fullfile(absolute_path_to_repo_root, 'CUDA/build/src/imwarp_flexible/Debug/run_imwarp_flexible.exe' )
    
    command_line = ...
        sprintf('%s %s %s %s %s %s %s', ...
                absolute_path_to_executable, ...
                'input.klb', ...
                'input_origin_and_spacing.txt', ...
                'affine_transform_matrix_in_y_equals_Ax_form_row_major.txt', ...
                'output_origin_and_spacing.txt', ...
                'output_dimensions.txt', ...
                'output.klb') ;
    fprintf('%s\n', command_line) ;        
    status = system(command_line) ;

    if status~=0 ,
        error('run_imwarp_flexible.exe failed with status %d', status) ;
    end

    output_stack = readKLBstack('output.klb') ;    
end
