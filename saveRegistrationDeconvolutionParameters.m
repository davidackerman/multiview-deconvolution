function saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameCell, psfFilenameCell, Tcell, verbose, deconvParam)

Nviews = length(imgFilenameCell);

fid = fopen(filenameXML, 'w');

%write header
fprintf(fid,'<?xml version="1.0" encoding="utf-8"?>\n');
fprintf(fid,'<document>\n');

for ii = 1:Nviews
    
    A = Tcell{ii};
    fprintf(fid,'<view imgFilename="%s" psfFilename="%s" A="%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f">', imgFilenameCell{ii}, psfFilenameCell{ii}, A(1), A(2), A(3), A(4),A(5), A(6), A(7), A(8),A(9), A(10), A(11), A(12),A(13), A(14), A(15), A(16));
    fprintf(fid,'</view>\n');
end


if( isfield(deconvParam,'prefix') )
   prefix = deconvParam.prefix; 
else
    prefix = '';
end

fprintf(fid,'<deconvolution lambdaTV="%f" numIter="%d" imBackground="%f" verbose="%d" blockZsize="%d" prefix="%s">', deconvParam.lambdaTV, deconvParam.numIter, deconvParam.imBackground, verbose, deconvParam.blockZsize, prefix);
fprintf(fid,'</deconvolution>\n');

%write footer
fprintf(fid,'</document>\n');

fclose(fid);