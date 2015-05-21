%we assume images are aligned and each PSF is transformed appropiately
function J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, numIters, lambdaTV, debug, saveIterBasename)

sigmaDer = 2.0;%smoothing to calculate the Gaussian derivatives
minWeightVal = 0.001;%to avoid multiplication by zero in weights
numIm = length(imCell);

stepWriteOutput = 5;

%normalize elements
J = zeros(size(imCell{1}));

if( ~isempty(weightsCell) )
    ww = ones(size(weightsCell{1}), 'single');
    for ii = 1:numIm
        weightsCell{ii}(weightsCell{ii} < minWeightVal) = minWeightVal;
        ww = ww .* single(weightsCell{ii});
    end
    
    ww = ww.^(1.0/numIm);%geometric mean
    
    for ii = 1:numIm
        weightsCell{ii} = weightsCell{ii} ./ ww;
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


%calculate "compund kernel": optimization I from Preibisch's paper
PSFcompound = PSFcell;
options.GPU = false;
options.Power2Flag = true;
for ii = 1:numIm
   
    %Hermitian
    psf = PSFcell{ii};
    for jj = 1:ndims(psf)
        psf = flip(psf,jj);
    end
    PSFcompound{ii} = ones(size(psf));
    %convolve with other views    
    for jj = 1:numIm
       if( ii ~= jj )
            PSFcompound{ii} = PSFcompound{ii} .* convnfft( psf, PSFcell{jj}, 'same',1:ndims(psf),options);
       end
    end
    %normalize
    PSFcompound{ii} = single(PSFcompound{ii}) / sum(PSFcompound{ii}(:));       
end

%start iterations
for iter = 1:numIters
    tic;    
    
    %precompute laplacian (even if we do sequential update
    if( lambdaTV > 0 )
        nl = normalizedLaplacian(J, sigmaDer);
    end
    
    
    %lucy-richardson
    aux = zeros(size(J),'single');
    for ii = 1:numIm
        aux = aux + stepLucyRichardson(imCell{ii},PSFcell{ii}, PSFcompound{ii}, J);
        if( isempty(weightsCell) )            
            aux = aux + stepLucyRichardson(imCell{ii},PSFcell{ii}, PSFcompound{ii}, J) / numIm;
        else
            aux = aux + stepLucyRichardson(imCell{ii},PSFcell{ii}, PSFcompound{ii}, J) .* weightsCell{ii};            
        end                        
    end
    %update final result
    J = J .* aux;
    %add total variation regularization: from Dey et al. "Richardson–Lucy Algorithm
    %With Total Variation Regularization for 3D Confocal Microscope
    %Deconvolution" 2006
    if( lambdaTV > 0 )
        J = J ./ (1.0 - lambdaTV * nl);
    end
    %normalize
    J = J / sum(J(:));
    
    %{
    %sequential lucy richardson: TODO so far I cannot make it converge (it becomes NAN very fast)
    for ii = 1:numIm
        aux = stepLucyRichardson(imCell{ii},PSFcell{ii}, PSFcompound{ii}, J);
        if( isempty(weightsCell) )            
            aux = aux / numIm;
        else
            aux = aux .* weightsCell{ii};            
        end        
        
        %update final result
        J = J .* aux;
        %add total variation regularization: from Dey et al. "Richardson–Lucy Algorithm
        %With Total Variation Regularization for 3D Confocal Microscope
        %Deconvolution" 2006
        if( lambdaTV > 0 )
            J = J ./ (1.0 - lambdaTV * nl);            
        end
        
        J = J / sum(J(:));
    end
    %}   
    
    
    if( debug > 0  && ndims(J) == 2)
       figure;
       imagesc(J);
       title(['Lucy Richardson iter =' num2str(iter)]);
    end
        
    
    if( ~isempty(saveIterBasename) && mod(iter,stepWriteOutput) == 0 )
       disp(['Writing iteration ' num2str(iter) '. Iter took ' num2str(toc) 'secs' ])
       writeKLBstack(single(J), [saveIterBasename num2str(iter,'%.5d') '.klb']); 
    end
end
