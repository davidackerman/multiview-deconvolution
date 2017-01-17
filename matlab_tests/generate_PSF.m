function [psf, spacing, FWHM] = generate_PSF(spacing, FWHM, n_sigmas)

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
    impulse = zeros( 2 * ceil(n_sigmas*sigma_in_voxels) + 1 ) ;
    impulse((numel(impulse) + 1)/2) = 1 ;

    %perform convolution
    psf = imgaussian_anisotropy(impulse, sigma_in_voxels) ;
end  % function



function I=imgaussian_anisotropy(I,sigma,siz)
    % IMGAUSSIAN filters an 1D, 2D color/greyscale or 3D image with an 
    % Gaussian filter. This function uses for filtering IMFILTER or if 
    % compiled the fast  mex code imgaussian.c . Instead of using a 
    % multidimensional gaussian kernel, it uses the fact that a Gaussian 
    % filter can be separated in 1D gaussian kernels.
    %
    % J=IMGAUSSIAN(I,SIGMA,SIZE)
    %
    % inputs,
    %   I: The 1D, 2D greyscale/color, or 3D input image with 
    %           data type Single or Double
    %   SIGMA: The sigma used for the Gaussian kernel
    %   SIZE: Kernel size (single value) (default: sigma*6)
    % 
    % outputs,
    %   J: The gaussian filtered image
    %
    % note, compile the code with: mex imgaussian.c -v
    %
    % example,
    %   I = im2double(imread('peppers.png'));
    %   figure, imshow(imgaussian(I,10));
    % 
    % Function is written by D.Kroon University of Twente (September 2009)

    if(~exist('siz','var')), siz=sigma*6; end


    ndimsI=sum(size(I)>1);
    if(length(sigma)~=ndimsI)
        error 'You must specify one sigma for each dimension of the image'
    end


    % Filter each dimension with the 1D Gaussian kernels\
    if(ndimsI==1)
        % Make 1D Gaussian kernel
        kk=1;
        x=-ceil(siz(kk)/2):ceil(siz(kk)/2);
        H = exp(-(x.^2/(2*sigma(kk)^2)));
        H = H/sum(H(:));

        I=imfilter(I,H, 'same' ,'replicate');
    elseif(ndimsI==2)
        % Make 1D Gaussian kernel
        kk=1;
        x=-ceil(siz(kk)/2):ceil(siz(kk)/2);
        H = exp(-(x.^2/(2*sigma(kk)^2)));
        H = H/sum(H(:));
        Hx=reshape(H,[length(H) 1]);


        kk=2;
        x=-ceil(siz(kk)/2):ceil(siz(kk)/2);
        H = exp(-(x.^2/(2*sigma(kk)^2)));
        H = H/sum(H(:));
        Hy=reshape(H,[1 length(H)]);
        I=imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
    elseif(ndimsI==3)


        if(size(I,3)<4) % Detect if 3D or color image
            % Make 1D Gaussian kernel
            kk=1;
            x=-ceil(siz(kk)/2):ceil(siz(kk)/2);
            H = exp(-(x.^2/(2*sigma(kk)^2)));
            H = H/sum(H(:));
            Hx=reshape(H,[length(H) 1]);

            % Make 1D Gaussian kernel
            kk=2;
            x=-ceil(siz(kk)/2):ceil(siz(kk)/2);
            H = exp(-(x.^2/(2*sigma(kk)^2)));
            H = H/sum(H(:));
            Hy=reshape(H,[1 length(H)]);
            for k=1:size(I,3)
                I(:,:,k)=imfilter(imfilter(I(:,:,k),Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
            end
        else
            % Make 1D Gaussian kernel
            kk=1;
            x=-ceil(siz(kk)/2):ceil(siz(kk)/2);
            H = exp(-(x.^2/(2*sigma(kk)^2)));
            H = H/sum(H(:));
            Hx=reshape(H,[length(H) 1 1]);

            % Make 1D Gaussian kernel
            kk=2;
            x=-ceil(siz(kk)/2):ceil(siz(kk)/2);
            H = exp(-(x.^2/(2*sigma(kk)^2)));
            H = H/sum(H(:));
            Hy=reshape(H,[1 length(H) 1]);

            % Make 1D Gaussian kernel
            kk=3;
            x=-ceil(siz(kk)/2):ceil(siz(kk)/2);
            H = exp(-(x.^2/(2*sigma(kk)^2)));
            H = H/sum(H(:));
            Hz=reshape(H,[1 1 length(H)]);

            I=imfilter(imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate'),Hz, 'same' ,'replicate');
        end
    else
        error('imgaussian:input','unsupported input dimension');
    end
end  % function
