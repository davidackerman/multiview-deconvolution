function [trimmed_transformed_PSF, transformed_PSF] = transform_and_trim_PSF_cpp(T_in_row_form, raw_PSF)
    % Determine absolute path to the .exe
    absolute_path_to_this_file = mfilename('fullpath') ;
    absolute_path_to_this_dir = fileparts(absolute_path_to_this_file) ;
    absolute_path_to_mvd_repo = fileparts(absolute_path_to_this_dir) ;
    absolute_path_to_exe_dir = fullfile(absolute_path_to_mvd_repo, 'CUDA/build/src/Debug') ;
    absolute_path_to_exe = fullfile(absolute_path_to_exe_dir, 'mvd_without_mvd.exe') ;
    
    n_views = 1 ;
    
    % Save all these PSFs to .klb files
    writeKLBstack(raw_PSF, 'random-raw-psf.klb') ;

    % Make up stack file names
    stack_file_names = repmat({'foo.klb'}, [1 n_views]) ;  % never read in test executable
    
    % Write a .smv.xml file to do things new-school
    Ts = {T_in_row_form} ;
    raw_psf_file_names = repmat({'random-raw-psf.klb'},[1 n_views]) ;
    output_smv_xml_file_name = sprintf('random-test-new-style.smv.xml') ;
    deconvolution_parameters = struct('blockZsize', {-1}, ...
                                      'imBackground', {100}, ...
                                      'lambdaTV', {1e-4}, ...
                                      'numIter', {15}, ...
                                      'verbose', {1}, ...
                                      'isPSFAlreadyTransformed', {0} ) ;
    save_smv_xml(output_smv_xml_file_name, stack_file_names, raw_psf_file_names, Ts, deconvolution_parameters) ;
    
    % Run the code the new way, with PSF-transforming done internally
    fprintf('\n\nRunning executable...') ;
    command_line = ...
        sprintf('%s %s %s', ...
                absolute_path_to_exe, ...
                'random-test-new-style.smv.xml') ;
    status = system(command_line) ;
    if status==0 ,
        fprintf('Run exited normally.\n') ;
    else
        error('.exe returned exit code %d', status) ;
    end

    file_name = sprintf('random-test-new-style.smv.xml_debug_psf_%d.klb', 0) ;
    trimmed_transformed_PSF = readKLBstack(file_name) ;

    file_name = sprintf('untrimmed_PSF_%d.klb', 0) ;
    transformed_PSF = readKLBstack(file_name) ;
    
    assert(isa(trimmed_transformed_PSF, 'single')) ;
    assert(isa(transformed_PSF, 'single')) ;    
end
