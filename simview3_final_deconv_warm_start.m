%parameters

debugBasename = ['T:\temp\deconvolution\20150505_185415_GCaMP6_TM000139\simview3_TM139_'] %where .mat files are located to read affine transformations

Nviews = 4;

numIterFinal = 40;    
numIterIni = 5;

%this has to agree with previous start
backgroundOffset = 100;
lambdaTV = 0.008;



%%
%loading data
imCell = cell(Nviews,1);
PSFcell = cell(Nviews,1);
weightsCell = cell(Nviews,1);

suffix = '.klb';
for ii = 1:Nviews
    disp(['Reading registered files to ' debugBasename '*Reg_' num2str(ii) suffix]);
    imCell{ii} = readKLBstack([debugBasename  'imReg_' num2str(ii) suffix]);
    PSFcell{ii} = readKLBstack([debugBasename   'psfReg_' num2str(ii) suffix]);
    weightsCell{ii} = readKLBstack([debugBasename  'weightsReg_' num2str(ii) suffix]);
end

%%
%run deconvolution
%perform lucy richardson
disp 'Calculating multi-view deconvolution...'
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, backgroundOffset, [numIterIni numIterFinal], lambdaTV, 0, [debugBasename 'LR_iter']);

