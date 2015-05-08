%we assume images are aligned and each PSF is transformed appropiately
function J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, numIters, lambdaTV, debug, saveIterBasename)

sigmaDer = 2.0;%smoothing to calculate the Gaussian derivatives

numIm = length(imCell);


%normalize elements
J = zeros(size(imCell{1}));
for ii = 1:numIm
   PSFcell{ii} = single(PSFcell{ii}) / sum(PSFcell{ii}(:)); 
   imCell{ii} = single(imCell{ii}) / sum(imCell{ii}(:)); 
   J = J + imCell{ii} / numIm;%initialize with average
end

%start iterations
for iter = 1:numIters
    tic;
    aux = zeros(size(J));
    
    %basic lucy richardson
    for ii = 1:numIm
        aux = aux + stepLucyRichardson(imCell{ii},PSFcell{ii}, J) / numIm;
    end
    
    
    
    %add total variation regularization: from Dey et al. "Richardson–Lucy Algorithm
    %With Total Variation Regularization for 3D Confocal Microscope
    %Deconvolution" 2006
    if( lambdaTV > 0 )        
        J = J .* aux ./ (1.0 - lambdaTV * normalizedLaplacian(J, sigmaDer));
        
    else%regular lucy richardson
        J = J .* aux;
    end
    
    
    
    
    if( debug > 0  && ndims(J) == 2)
       figure;
       imagesc(J);
       title(['Lucy Richardson iter =' num2str(iter)]);
    end
    
    J = J / sum(J(:));
    
    if( ~isempty(saveIterBasename) )
       disp(['Writing iteration ' num2str(iter) '. Iter took ' num2str(toc) 'secs' ])
       size(J)
       fid = fopen([saveIterBasename num2str(iter,'%.5d') '.raw'],'wb'); 
       fwrite(fid, single(J(:)), 'float32');
       fclose(fid);
    end
end
