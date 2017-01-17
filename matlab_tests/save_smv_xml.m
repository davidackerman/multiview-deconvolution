function save_smv_xml(smv_xml_file_name, stack_file_names, psf_file_names, As, deconvolution_parameters)
    % Note that As{i} is assumed to be in imwarp() form, i.e. row form
    
    n_views = length(stack_file_names);

    fid = fopen(smv_xml_file_name, 'w') ;

    % write header
    fprintf(fid,'<?xml version="1.0" encoding="utf-8" standalone="no"?>\n');
    fprintf(fid,'<document>\n');

    for i_view = 1:n_views    
        A = As{i_view};
        % We write A out in col-major order, b/c MVD expects a row-form
        % transform matrix in col-major order.
        fprintf(fid, ...
                '<view A="%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f" imgFilename="%s" psfFilename="%s"/>\n',  ...
                A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8),A(9), A(10), A(11), A(12), A(13), A(14), A(15), A(16), stack_file_names{i_view}, psf_file_names{i_view});
        %fprintf(fid,'</view>\n');
    end


    % if( isfield(deconvParam,'prefix') )
    %    prefix = deconvParam.prefix; 
    % else
    %     prefix = '';
    % end
    % 
    % 
    % if( ~isfield(deconvParam,'saveAsUINT16') )
    %     deconvParam.saveAsUINT16 = 1;%default value
    % end
    % 
    % if( ~isfield(deconvParam,'weightThreshold') )
    %     deconvParam.weightThreshold = 0.05;%default value
    % end



    if isfield(deconvolution_parameters, 'isPSFAlreadyTransformed') ,
        fprintf(fid, ...
                '<deconvolution blockZsize="%d" imBackground="%f" lambdaTV="%f" numIter="%d" saveAsUINT16="%d" verbose="%d" isPSFAlreadyTransformed="%d"/>\n', ...
                deconvolution_parameters.blockZsize, ...
                deconvolution_parameters.imBackground, ...
                deconvolution_parameters.lambdaTV, ...
                deconvolution_parameters.numIter, ...
                deconvolution_parameters.saveAsUINT16, ...                
                deconvolution_parameters.verbose, ...
                deconvolution_parameters.isPSFAlreadyTransformed);
    else
        fprintf(fid, ...
                '<deconvolution blockZsize="%d" imBackground="%f" lambdaTV="%f" numIter="%d" saveAsUINT16="%d" verbose="%d"/>\n', ...
                deconvolution_parameters.blockZsize, ...
                deconvolution_parameters.imBackground, ...
                deconvolution_parameters.lambdaTV, ...
                deconvolution_parameters.numIter, ...
                deconvolution_parameters.saveAsUINT16, ...                
                deconvolution_parameters.verbose);
    end        
    %fprintf(fid,'</deconvolution>\n');

    %write footer
    fprintf(fid,'</document>\n');

    fclose(fid);

end
