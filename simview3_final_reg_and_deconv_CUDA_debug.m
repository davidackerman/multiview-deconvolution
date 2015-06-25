%parameters

debugBasename = ['C:\Users\Fernando\matlabProjects\deconvolution\CUDA\test\data'] %where .mat files are located to read affine transformations

Nviews = 4;

numItersLR = 15;
backgroundOffset = 100;
lambdaTV = -1.0;



%%
%loading data
imCell = cell(Nviews,1);
PSFcell = cell(Nviews,1);
weightsCell = cell(Nviews,1);
for ii = 1:Nviews
    disp(['Reading registered files to ' debugBasename '*Reg_' num2str(ii) '.klb']);
    imCell{ii} = readKLBstack([debugBasename filesep 'imReg_' num2str(ii) '.klb']);
    PSFcell{ii} = readKLBstack([debugBasename filesep  'psfReg_' num2str(ii) '.klb']);
    weightsCell{ii} = readKLBstack([debugBasename filesep 'weightsReg_' num2str(ii) '.klb']);
end
%%
%run deconvolution
%perform lucy richardson
disp 'Calculating multi-view deconvolution...'
J = multiviewDeconvolutionLucyRichardson(imCell,PSFcell, weightsCell, backgroundOffset, numItersLR, lambdaTV, 0, [debugBasename filesep 'Matlab_LR_iter']);

