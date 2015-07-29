TM = 1;
TMstr = num2str(TM,'%.6d');

xmlFilename = ['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM' TMstr '_Fiji\dataset.xml']

imPath = ['S:\SiMView3\15-06-11\Dme_E2_His2AvRFP_spiderGFP_12-03_20150611_155054.corrected\SPM00\TM' TMstr '\']
imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN02.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN02.klb']};%0,90,180,270 degrees

transposeOrigImage = true;

viewNumber = [1 2];

%cropping purposes
ROI = [129 333  1;...
       704 1628 500];

%%
%read transformations
tformFiji = importAffineTransformationsFiji(xmlFilename);
anisotropyZ = tformFiji{1}{end}(3,3);%calibration transformation


%%
%apply transformation
F = eye(4);
F(1:2,1:2) = [0 1;1 0];

%we need ot center image
B = eye(4);
B(4,1:3) =  [0 0 0];
C = eye(4);
C(4,1:3) = -B(1:3,4);

for ii = viewNumber

    %read image
    imFilename = [imPath imFilenameCell{ii}];
    im = readKLBstack(imFilename);
    
    if( transposeOrigImage )
        im = permute(im, [2 1 3]);
    end
    
    imRefSize = size(im) .* [ 1 1 anisotropyZ];  
    
    %calculate transformation
    A = eye(1);
    for jj = 1:length(tformFiji{ii})
       A = tformFiji{ii}{jj} * A;
    end
    
    A = F \ (A*F);
    A = C * A * B;
    A
    
    %apply tranformation
    addpath './imWarpFast/'
    imW = imwarpfast(im, A, 0, imRefSize);
    rmpath './imWarpFast/'
    
    %crop output
    if ( isempty(ROI) == false )
        imW = imW(ROI(1,1):ROI(2,1), ROI(1,2):ROI(2,2), ROI(1,3):ROI(2,3));
    end

    %write out file
    writeKLBstack(imW, ['E:\temp\debug_imWarp_Fiji_view' num2str(ii-1) '.klb']);
end