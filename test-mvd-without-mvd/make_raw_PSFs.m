% Load the old-style XML
[Ts, stack_file_names, psf_file_names] = load_smv_xml('test.smv.xml') ;

% Write out a raw PSF for each view
n_views = length(Ts) ;
raw_psf_file_names = cell(1, n_views) ;
for i_view = 1:n_views ,
    % Load the transformed, trimmed PSF
    desired_trimmed_transformed_psf_file_name = sprintf('psf-view-%d.klb', i_view-1) ;
    desired_trimmed_transformed_PSF = readKLBstack(desired_trimmed_transformed_psf_file_name) ;
    
    % Compute the raw transform for the current view
    T = Ts{i_view} ;
    raw_PSF = untransform_psf(desired_trimmed_transformed_PSF, T) ;
    
    % Save the untransformed PSF to a .klb file
    raw_psf_file_name = sprintf('raw-psf-view-%d.klb', i_view-1) ;
    writeKLBstack(single(raw_PSF), raw_psf_file_name) ;
    
    % Save the name of the raw PSF file
    raw_psf_file_names{i_view} = raw_psf_file_name ;
end

% Output an SMV XML file for the raw PSFs and the original images
output_smv_xml_file_name = sprintf('test-with-psf-transforming.smv.xml') ;
deconvolution_parameters = struct('blockZsize', {-1}, ...
                                  'imBackground', {100}, ...
                                  'lambdaTV', {1e-4}, ...
                                  'numIter', {15}, ...
                                  'verbose', {1}, ...
                                  'isPSFAlreadyTransformed', {0} ) ;
save_smv_xml(output_smv_xml_file_name, stack_file_names, raw_psf_file_names, Ts, deconvolution_parameters) ;
