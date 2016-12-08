function [output_dims, output_origin] =  run_transform_lattice_cpp(T_in_row_form, ...
                                                                   input_dims, input_origin, input_spacing, ...
                                                                   output_spacing)

    input_dims_int64 = int64(input_dims) ;
    
    write_as_raw('T_in_row_form.raw', T_in_row_form) ;
    write_as_raw('input_dims.raw', input_dims_int64) ;
    write_as_raw('input_origin.raw', input_origin) ;
    write_as_raw('input_spacing.raw', input_spacing) ;
    write_as_raw('output_spacing.raw', output_spacing) ;
        
%     % Check that we can read those things back in
%     T_in_row_form_check = read_as_raw('T_in_row_form.raw', 4, 4, 'double') ;
%     input_dims_int64_check = read_as_raw('input_dims.raw', 1, 3, 'int64') ;
%     input_origin_check = read_as_raw('input_origin.raw', 1, 3, 'double') ;
    
    % Get the executable name
    absolute_path_to_this_file = mfilename('fullpath') ;
    absolute_path_to_repo_root = fileparts(fileparts(absolute_path_to_this_file)) ;
    absolute_path_to_executable = fullfile(absolute_path_to_repo_root, 'CUDA/build/src/Debug/run_transform_lattice.exe' ) ;
    
    command_line = ...
        sprintf('%s %s %s %s %s %s %s %s', ...
                absolute_path_to_executable, ...
                'T_in_row_form.raw', ...
                'input_dims.raw', ...
                'input_origin.raw', ...
                'input_spacing.raw', ...
                'output_spacing.raw', ...
                'output_dims.raw', ...
                'output_origin.raw') ;
    fprintf('%s\n', command_line) ;        
    status = system(command_line) ;

    if status~=0 ,
        error('run_transform_lattice.exe failed with status %d', status) ;
    end

    output_dims_int64 = read_as_raw('output_dims.raw', 1, 3, 'int64') ;
    output_origin = read_as_raw('output_origin.raw', 1, 3, 'double') ;

    output_dims = double(output_dims_int64) ;
end
