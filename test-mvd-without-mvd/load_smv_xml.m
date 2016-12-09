function [As, stackFileNames, psfFileNames] = load_smv_xml(file_name)
    % Note that the A's returned by this are in the form that imwarp()
    % wants, which is the transpose of the seems-sensible-to-me way to do it.
    
    fid = fopen(file_name, 'r') ;
    n_views_so_far = 0 ;
    As = cell(1,0) ;
    stackFileNames = cell(1,0) ;
    psfFileNames = cell(1,0) ;

    tline = fgetl(fid);
    while ischar(tline) ,
        if( length(tline) >= 5 && strcmp(tline(1:5),'<view') == 1)
            n_views_so_far = n_views_so_far + 1;

            % Read the transform matrix
            pp = findstr(tline,'A="') + 2;
            pp2 = findstr(tline,'"');        
            idx = find(pp2 == pp);
            As{1,n_views_so_far} = reshape(str2num(tline(pp+1:pp2(idx+1)-1)), [4 4]) ;  % this is in imwarp() form.

            % Read the stack file name
            p1 = findstr(tline,'imgFilename="');
            aux = tline(p1 + 13:end);
            p2 = findstr(aux,'"');
            stackFileNames{1,n_views_so_far} = aux(1:p2(1)-1) ;

            % Read the PSF stack file name
            p1 = findstr(tline,'psfFilename="');
            aux = tline(p1 + 13:end);
            p2 = findstr(aux,'"');
            psfFileNames{1,n_views_so_far} = aux(1:p2(1)-1) ;

        end

        % Read the next line
        tline = fgetl(fid);
    end

    fclose(fid);
    
end
