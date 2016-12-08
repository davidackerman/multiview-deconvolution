function [psf, spacing, FWHM] = generate_PSF_alternate(spacing, FWHM, n_sigmas)

if ~exist('spacing', 'var') || isempty(spacing) ,
    spacing = [0.40625 0.40625 2.03125] ;  % um, in order y x z
end

if ~exist('FWHM', 'var') || isempty(FWHM) , 
    FWHM = [1.0 1.0 4.5] ;  % um, in order y x z
end

if ~exist('n_sigmas', 'var') || isempty(n_sigmas) , 
    n_sigmas = 8 ;  % um
end

% sigma = zeros(1,2) ;
% for ii = 1:3
%    sigma(ii) = sqrt( -0.5*((FWHM(ii)/sampling(ii))^2)/log(0.5));     
% end
sigma_in_voxels = sqrt( -0.5*((FWHM./spacing).^2)/log(0.5)) ;   % should be 3x1

% Make a stack with just an impulse in the center
max_sigma_in_voxels = max(sigma_in_voxels) ;
radius_in_voxels = ceil(n_sigmas*max_sigma_in_voxels) ;
diameter_in_voxels = 2*radius_in_voxels+1 ;
impulse = zeros(repmat(diameter_in_voxels,[1 3])) ;
impulse((numel(impulse) + 1)/2) = 1 ;

%perform convolution
psf = imgaussian_anisotropy(impulse, sigma_in_voxels) ;
