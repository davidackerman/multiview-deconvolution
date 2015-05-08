function PSF = generatePSF(sampling,FWHMpsf, filenameOut)

if( isempty( sampling) )
    sampling = [0.40625 0.40625 5.2];%in um
end

if( isempty(FWHMpsf) )
    FWHMpsf = [0.8 0.8 6.0];%in um
end

sigma = zeros(1,2);
for ii = 1:3
   sigma(ii) = sqrt( -0.5*((FWHMpsf(ii)/sampling(ii))^2)/log(0.5));     
end

PSF = zeros(2 * ceil(8 *sigma) + 1 );

%perform convolution
PSF((numel(PSF) + 1)/2) = 1.0;
PSF = imgaussianAnisotropy(PSF,sigma);


if( ~isempty(filenameOut) )
   fid = fopen(filenameOut,'wb');
   fwrite(fid,single(PSF), 'float32');
   fclose(fid);
end






