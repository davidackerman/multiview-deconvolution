%script to test expected resolution after deconvolution
%we assume 1 pixel = PixexSampling nm


for rrSq = 3000: 500: 15000
    
    pixelSampling = 100;
    
    rotDeg = 0;%rotation of the "squared" points in order to analyze different directions
    radiusSquare = rrSq / pixelSampling; %separation in pixels
    radiusSource = 200 / pixelSampling;%radius of the spot in pixels
    
    
    noiseSigma = 3.2;%additive Gaussian noise
    noiseMu = 100;%offset
    numPhotons = 10000;%average number of photons emitted from the point source
    
    sigmaPSF_z = 6000 / pixelSampling;%PSF on the optical axis direction (worse one) in pixels
    
    debug = 0;
    %%
    %generate PSF
    numViews = 2;
    PSFcell = cell(numViews,1);
    
    sigma = [500/pixelSampling sigmaPSF_z; sigmaPSF_z 500/pixelSampling];
    
    for ii = 1:numViews
        v = fspecial('gaussian', [6*sigma(ii,1), 1], sigma(ii,1) ); % vertical filter
        h = fspecial('gaussian', [1, 6*sigma(ii,2)], sigma(ii,2) ); % horizontal
        PSFcell{ii} = v*h;
        PSFcell{ii} = PSFcell{ii};
    end
    
    %%
    %setup points
    mu = [1 0; 0 1;-1 0;0 -1];
    
    %apply rotation
    R = [cosd(rotDeg) -sind(rotDeg);sind(rotDeg) cosd(rotDeg)];
    mu = (R * mu' )';
    
    %apply scaling
    mu = mu * radiusSquare;
    
    imOrig = zeros(ceil(max(abs(mu)) + radiusSource + 10 * sigmaPSF_z + 20));
    [XI, YI] = ndgrid(1:size(imOrig,1),1:size(imOrig,2));
    cc = (size(imOrig)+1) / 2;
    
    for ii = 1:size(mu,1)
        rr = sqrt((XI - cc(1) - mu(ii,1)).^2 + (YI - cc(2) - mu(ii,2)).^2);
        imOrig(rr < radiusSource ) = numPhotons;
    end
    
    %add poisson noise
    imOrig = poissrnd(imOrig);
    
    
    %display result
    if( debug > 0 )
        figure;
        imagesc(imOrig);
        title('Original image');
    end
    
    %%
    %generate blurs from different views
    imCell = PSFcell;
    for ii = 1:numViews
        imCell{ii} = single(convn(imOrig,PSFcell{ii},'same')) + noiseSigma * randn(size(imOrig)) + noiseMu;
        if( debug > 0 )
            figure;
            imagesc( imCell{ii});
            title(['Blurred view ' num2str(ii)]);
        end
    end
    
    %%
    %apply deconvolution
    numIters = 20;
    J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, numIters, debug);
    
    %display final result
    if( debug == 0 )
        figure;imagesc(J);
        title(['Deconvolved image for radius' num2str(radiusSquare)]);
    end
    
    
end



