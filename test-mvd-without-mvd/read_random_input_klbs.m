n_views = 3 ;

% Read them in, compare
fprintf('Reading in old-style debug PSFs...\n') ;
old_style_psf = cell(1,n_views) ;
for i_view = 1:n_views
    i_view
    file_name = sprintf('random-test-old-style.smv.xml_debug_psf_%d.klb', i_view-1) ;
    old_style_psf{i_view} = readKLBstack(file_name) ;
end

fprintf('Reading in new-style debug PSFs...\n') ;
new_style_psf = cell(1,n_views) ;
for i_view = 1:n_views
    i_view
    file_name = sprintf('random-test-new-style.smv.xml_debug_psf_%d.klb', i_view-1) ;
    new_style_psf{i_view} = readKLBstack(file_name) ;
end

% Compare them
fprintf('Comparing old- and new-style debug PSFs...\n') ;
for i_view = 1:n_views ,
    ice = is_close_enough(new_style_psf{i_view}, old_style_psf{i_view}) ;
    if ~ice ,
        fprintf('Too-large deviation in view %d, using one-based indexing!!!!\n', i_view) ;
    end
end
