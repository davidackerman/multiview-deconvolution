sizeVec = [128 256 512 1024];
numItersVec = [20 40 80 160];

%%
pathFiles = 'B:\keller\deconvolution\synthetic_dataset\'
pathFilesCluster = '/nobackup/keller/deconvolution/synthetic_dataset/';

samplingXYZ = [0.40625, 0.40625, 3.0];%in um
FWHMpsf = [0.8, 0.8, 4.0]; %theoretical full-width to half-max of the PSF in um.


deconvParam.lambdaTV = 0.000; %Ttotal variation regularization lambda parameter. Set to < 0 to deactivate total variation regularization
%deconvParam.numIter = 40; %number of Lucy-Richardson iterations
deconvParam.imBackground = 100.0; %image background level. It will be subtracted from images to make sure noise statistics are as Poissonian as possible
deconvParam.blockZsize = -1; %for large datasets (i.e. mouse) to calculate deconvolution using multiple z blocks. Set to <0 to not split data. Otherwise set to a power of 2.
deconvParam.prefix = '';%we will use this for PSF differentiation

verbose = 0;


%%
%calculate fix parameters
Nviews = 4;
anisotropyZ = samplingXYZ(3)/ samplingXYZ(1);
A = eye(4);
A(3,3) = anisotropyZ;

Acell = cell(1,Nviews);
for ii = 1:Nviews
    Acell{ii} = A;
end

%generate PSF files
psfFilenameCell = PSFcell;
psfFilenameClusterCell = PSFcell;
for ii = 1:Nviews    
    %save psf
    psfFilenameCell{ii} = [pathFiles 'psfFiles\psfReg_' deconvParam.prefix '_view' num2str(ii) '.klb'];   
    psfFilenameClusterCell{ii} = [pathFilesCluster 'psfFiles/psfReg_' deconvParam.prefix '_view' num2str(ii) '.klb'];
end
PSFcell = generateTransformedPSF(samplingXYZ, FWHMpsf,Acell,psfFilenameCell);

%%
count = 0;
for size_ = sizeVec
    
    imSize = [size_ size_ floor(size_/anisotropyZ)];
    deconvParam.prefix = ['imSize_' num2str(size_)];
    
    %generate image
    im = reshape([1:prod(imSize)],imSize);
    im = mod(im, 2^16);
    im = imgaussian(im, 3.0, 6.0);
    im = uint16(im + 3.2 * randn(imSize) + 100);
    
    
    imgFilenameClusterCell = cell(Nviews,1);
    imgFilenameCell = cell(Nviews,1);
    for ii = 1:Nviews
        imgFilenameCell{ii} = [pathFiles 'imReg_' deconvParam.prefix '_view' num2str(ii) '.klb'];
        writeKLBstack(im, imgFilenameCell{ii});
        imgFilenameClusterCell{ii} = [pathFilesCluster 'imReg_' deconvParam.prefix '_view' num2str(ii) '.klb'];
    end
    
    for numIters = numItersVec
        deconvParam.numIter = numIters;
        %save deconvolition XML parameters
        count = count + 1;
        filenameXML = [pathFiles 'xmlFiles\MVparam_' num2str(count) '.xml'];
        saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameClusterCell, psfFilenameClusterCell, Acell, verbose, deconvParam);
    end
end

%%
%process results
count = 0;

param = -ones(1,3);%imSize, numIters; time(ms)
for size_ = sizeVec
    for numIters = numItersVec
        count = count + 1;
        param(count,[1 2]) = [size_ numIters];
    end
end

%read files
str_ = 'Calculating multiview deconvolution';
D = dir([pathFiles 'outputFiles\output_synthetic_*']);
for ii = 1:length(D)
    fid = fopen([pathFiles 'outputFiles\' D(ii).name],'r');
    while(1)
        tline = fgetl(fid);
        if ~ischar(tline), break, end
        
        if( length(tline) > length(str_) && strcmp(tline(1:length(str_)),str_) == 1)
            while(1)
                tline = fgetl(fid);
                if ~ischar(tline), break, end;
                if( length(tline) >= 4 && strcmp(tline(1:4),'Took') == 1)
                    p1 = strfind(tline,'ms');
                    param(ii,3) = str2double(tline(5:p1-1));
                    %[ii param(ii,3)]
                    break;
                end
            end
            
        end
    end
    
    fclose(fid);
end

%%
%plot results
figure;
color_ = {'m', 'c', 'k', 'g'};
count = 0;
for numIters = numItersVec
    count = count + 1;
    pp = param(:,2) == numIters;
    hold on;
    loglog(sizeVec.^3 / 2^20,param(pp,3) / 1000, color_{count});
    hold off;
end
xlabel('Image size (Megavoxels)')
ylabel('GPU time (including I/O) (secs)');
title('Multi-view deconvolution Multi-GPU code (4 views)');
legend('20 iters','40 iters','80 iters','160 iters','location', 'best');
