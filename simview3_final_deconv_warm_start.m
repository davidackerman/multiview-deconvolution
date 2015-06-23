%parameters

debugBasename = ['T:\temp\deconvolution\20150522_160119_fly_with_beads_TM000002\simview3_TM2_'] %where .mat files are located to read affine transformations

Nviews = 4;

numIterFinal = 200;    
numIterIni = 100;

%this has to agree with previous start
backgroundOffset = 100;
lambdaTV = 0.008;



%%
%loading data
imCell = cell(Nviews,1);
PSFcell = cell(Nviews,1);
weightsCell = cell(Nviews,1);

suffix = '.raw';
for ii = 1:Nviews
    disp(['Reading registered files to ' debugBasename '*Reg_' num2str(ii) suffix]);
    imCell{ii} = readRawStack([debugBasename  'imReg_' num2str(ii) suffix]);
    PSFcell{ii} = readRawStack([debugBasename   'psfReg_' num2str(ii) suffix]);
    weightsCell{ii} = readRawStack([debugBasename  'weightsReg_' num2str(ii) suffix]);
end

%load last LR iteration

%%
%run deconvolution
%perform lucy richardson
disp 'Calculating multi-view deconvolution...'
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, backgroundOffset, [numIterIni numIterFinal], lambdaTV, 0, [debugBasename 'LR_iter']);

