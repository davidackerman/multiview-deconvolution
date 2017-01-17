function PSF = generateUntransformedPSF(sampling, FWHMpsf, psfFilename)

PSF = generatePSF(sampling, FWHMpsf, []) ;

% Nviews = length(Acell);
% 
% PSFcell = cell(Nviews,1);
% for ii = 1:Nviews
%     %apply transformation
%     PSFaux = single(imwarp(PSF, affine3d(Acell{ii}), 'interp', 'cubic'));
%     
%     %make sure it does not have negative values from cubic "ringing"
%     PSFaux( PSFaux < 0 ) = 0;
%     
%     %crop PSF to reduce it in size
%     PSFcell{ii} = trimPSF(PSFaux, 1e-10);
%     
%     %normalize PSF
%     PSFcell{ii} = PSFcell{ii} / sum(PSFcell{ii}(:));
%     
%     %save psf
%     if( nargin > 3 )
%         writeKLBstack(PSFcell{ii}, psfFilenameCell{ii}, -1, [],[],0,[]);
%     end
% end

% Save untransformed PSF
if nargin > 2 ,
    writeKLBstack(PSF, psfFilename, -1, [], [], 0, []) ;
end
