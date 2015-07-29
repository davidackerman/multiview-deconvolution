%parameter

TM = 2899;
filePath = 'T:\temp\deconvolution\20150505_125300_GCaMP6_TM002899\'

pathImPattern = ['S:\SiMView3\15-05-05\Dme_L1_57C10-GCaMP6s_20150505_125300.corrected\SPM00\TM??????\SPM00_TM??????_' ];
imSuffix = {'CM00_CHN01', 'CM02_CHN00', 'CM01_CHN01', 'CM03_CHN00'};
PSFfilename = 'PSF_synthetic.klb';

verbose = 1;

deconvParam.lambdaTV = 0.0001; 
deconvParam.imBackground = 100.0;
deconvParam.numIter = 20;


filenameXML = [filePath 'regDeconvParam.xml'];

%%
%load coarse registration
coarse = load([filePath   'imRegister_Matlab_tform.mat'],'tformCell','imPath','imFilenameCell', 'numLevels', 'anisotropyZ');

%load fine registration
fine = load([filePath   'imWarp_Matlab_tform_fine.mat'],'tformCell', 'Tcell', 'imPath', 'imFilename');

%%
Nviews = length(imSuffix);

imgFilenameCell = cell(Nviews,1);
psfFilenameCell = cell(Nviews,1);
Tcell = cell(Nviews,1);

for ii = 1:Nviews
    filename = recoverFilenameFromPattern(pathImPattern,TM);
    imgFilenameCell{ii} = [filename imSuffix{ii} '.klb'];
   
    %afine transformation
    Tcell{ii} = coarse.tformCell{ii} * fine.tformCell{ii};
    
    %generate PSF
    PSF = readKLBstack([filePath PSFfilename]);
    PSF = imwarp(PSF, affine3d(Tcell{ii}), 'interp', 'cubic');
    psfFilenameCell{ii} = [filePath 'psfReg_view' num2str(ii) '.klb'];
    writeKLBstack(single(PSF), psfFilenameCell{ii});
end

%%
%save XML
saveRegistrationDeconvolutionParameters(filenameXML,imgFilenameCell, psfFilenameCell, Tcell, verbose, deconvParam)