%we assume images are aligned and each PSF is transformed appropiately
function J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, numIters, lambdaTV, debug, saveIterBasename)

sigmaDer = 2.0;%smoothing to calculate the Gaussian derivatives

numIm = length(imCell);


%normalize elements
J = zeros(size(imCell{1}));

if( ~isempty(weightsCell) )
    ww = single(weightsCell{1});
    for ii = 2:numIm
        ww = ww + single(weightsCell{ii});
    end
    
    ww( ww < eps('single') ) = inf;%this pixel is dead from all views
    for ii = 1:numIm
        weightsCell{ii} = weightsCell{ii} / ww;
    end
    clear ww;
end
for ii = 1:numIm
   PSFcell{ii} = single(PSFcell{ii}) / sum(PSFcell{ii}(:)); 
   imCell{ii} = single(imCell{ii}) / sum(imCell{ii}(:));
   
   if( isempty(weightsCell) )
       J = J + imCell{ii} / numIm;%initialize with average
   else
       J = J + weightsCell{ii} .* imCell{ii};%initialize with weighted average
   end
end

%start iterations
for iter = 1:numIters
    tic;
    aux = zeros(size(J));
    
    %basic lucy richardson
    if( isempty(weightsCell) )
        for ii = 1:numIm
            aux = aux + stepLucyRichardson(imCell{ii},PSFcell{ii}, J) / numIm;
        end
    else
        for ii = 1:numIm
            aux = aux + stepLucyRichardson(imCell{ii},PSFcell{ii}, J) .* weightsCell{ii};
        end
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
