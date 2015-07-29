function script_saveStacksForFiji_15_06_11_JFRCcluster(TM)
%%

transposeOrigImage = true;
samplingXYZ = [0.40625, 0.40625, 3.250];%in um

TMstr = num2str(TM,'%.6d');
imPath = ['S:\SiMView3\15-06-11\Dme_E2_His2AvRFP_spiderGFP_12-03_20150611_155054.corrected\SPM00\TM' TMstr '\']

weightsScaling = 65535.0;
%%
%map network drive for JFRC cluster
networkFolder = 'B:\keller\15_06_11_fly_functionalImage';
if(exist(networkFolder) == 0)
    %read password
    fid = fopen('HHMIcredentials.txt','r');
    username = fgetl(fid);
    password = fgetl(fid);
    fclose(fid);
    %mount networkd drive
    cmd = ['net use b: \\fxt.int.janelia.org\nobackup /user:' username ' ' password];
    [rr,ss] = system(cmd);
else
    rr = -1;
    ss = 'Network drive already existed';
end


%%
TMstr = num2str(TM,'%.6d');

for isMembrane = [0 1]
    
    if( isMembrane == 0 )
        %for nuclear channel
        imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN02.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN02.klb']};%0,90,180,270 degrees
        outputFolder = [networkFolder '\TM' TMstr '_Fiji\'];
    else
        %for membrane channel
        imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN03.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN03.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees
        outputFolder = [networkFolder '\TM' TMstr 'membrCh_Fiji\'];
    end
    %%
    if( exist(outputFolder) == 0 )
        mkdir(outputFolder);
    end
    
    %%
    %save files as tiff
    for ii = 1:length(imFilenameCell)
        filename = [imPath imFilenameCell{ii}];
        im = readKLBstack(filename);
        
        if( transposeOrigImage )
            im = permute(im, [2 1 3]);
        end
        
        
        %apply basic transformation
        if( ii == 1)
            anisotropyZ = samplingXYZ(3) / samplingXYZ(1);
            imRefSize = ceil(size(im) .* [1 1 anisotropyZ]);
        end
        
        %write image
        writeTifStack(im,[outputFolder 'simview3_view' num2str(ii-1)]);
        
        
        %calculate weigths
        disp('Calculating contrast weights');
        imW = single(estimateDeconvolutionWeights(im, anisotropyZ , 15, []));
        
        imW = imW * weightsScaling;%imwrite does not save tiff 32
        writeTifStack(uint16(imW),[outputFolder 'simview3_weights_view' num2str(ii-1)]);
        
        %copy PSF
        copyfile(['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM000720_Fiji\simview3_TM720_psfReg_' num2str(ii) '.klb'],[outputFolder 'simview3_TM720_psfReg_' num2str(ii) '.klb']);
        
    end
    %copy XML files
    copyfile(['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM000720_Fiji\dataset.xml'],[outputFolder 'dataset.xml']);
    copyfile(['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM000720_Fiji\dataset_weights.xml'],[outputFolder 'dataset_weights.xml']);
end

