function [psf, spacing, FWHM] = generate_PSF_with_given_dims(spacing_xyz, FWHM_xyz, dims_yxz)

if ~exist('spacing_xyz', 'var') || isempty(spacing_xyz) ,
    spacing = [0.40625 0.40625 2.03125] ;  % um, in order x y z
end

if ~exist('FWHM_xyz', 'var') || isempty(FWHM_xyz) , 
    FWHM = [1.0 1.0 4.5] ;  % um, in order x y z
end

if iscolumn(spacing_xyz) ,
    spacing_xyz = spacing_xyz' ;
end

% sigma = zeros(1,2) ;
% for ii = 1:3
%    sigma(ii) = sqrt( -0.5*((FWHM(ii)/sampling(ii))^2)/log(0.5));     
% end
sigma_xyz_in_voxels = sqrt( -0.5*((FWHM_xyz./spacing_xyz).^2)/log(0.5)) ;   % should be 3x1
sigma_yxz_in_voxels = sigma_xyz_in_voxels * [0 1 0 ; 1 0 0 ; 0 0 1] ;

% Make a stack with just an impulse in the center
%max_sigma_in_voxels = max(sigma_in_voxels) ;
%radius_in_voxels = ceil(n_sigmas*max_sigma_in_voxels) ;
%diameter_in_voxels = 2*radius_in_voxels+1 ;
impulse = zeros(dims_yxz) ;
impulse((numel(impulse) + 1)/2) = 1 ;

%perform convolution
psf = imgaussian_anisotropy(impulse, sigma_yxz_in_voxels, dims_yxz) ;
