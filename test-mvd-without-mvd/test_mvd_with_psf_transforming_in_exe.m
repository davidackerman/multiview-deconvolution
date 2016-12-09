% Determine absolute path to the .exe
absolute_path_to_this_file = mfilename('fullpath') ;
absolute_path_to_this_dir = fileparts(absolute_path_to_this_file) ;
absolute_path_to_mvd_repo = fileparts(absolute_path_to_this_dir) ;
absolute_path_to_exe_dir = fullfile(absolute_path_to_mvd_repo, 'CUDA/build/src/Debug') ;
absolute_path_to_exe = fullfile(absolute_path_to_exe_dir, 'mvd_without_mvd.exe') ;

% Run the code the usual way, with PSF-transforming done externally
command_line = ...
    sprintf('%s %s %s', ...
            absolute_path_to_exe, ...
            'test.smv.xml') ;
status = system(command_line) ;

% Run the code the new way, with PSF-transforming done internally
command_line = ...
    sprintf('%s %s %s', ...
            absolute_path_to_exe, ...
            'test-with-psf-transforming.smv.xml') ;
status = system(command_line) ;

% Read in the transformed PSFs saved as debug output when running
% test.smv.xml

n_views = 3 ;
old_style_psf = cell(1,n_views) ;
for i_view = 1:n_views
    file_name = sprintf('test.smv.xml_debug_psf_%d.klb', i_view-1) ;
    old_style_psf{i_view} = readKLBstack(file_name) ;
end

new_style_psf = cell(1,n_views) ;
for i_view = 1:n_views
    file_name = sprintf('test-with-psf-transforming.smv.xml_debug_psf_%d.klb', i_view-1) ;
    new_style_psf{i_view} = readKLBstack(file_name) ;
end

% Compare them
for i_view = 1:n_views ,
    ice = is_close_enough(new_style_psf{i_view}, old_style_psf{i_view}) ;
    if ~ice ,
        error('Deviation!! Deviation!!') ;
    end
end
