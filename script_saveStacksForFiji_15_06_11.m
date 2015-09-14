%%
%parameters
TMvec = [1,400, 600, 720, 900, 1150, 1600]
%TMvec = 400
transposeOrigImage = true;
samplingXYZ = [0.40625, 0.40625, 3.250];%in um
angles = [0 90 180 270];

isMembraneVec = [0 1];

%%
%matrix flip xy coordinates
F = eye(4);
F(1:2,1:2) = [0 1;1 0];

for isMembrane = isMembraneVec
    %main loop
    for TM = TMvec
        
        
        TMstr = num2str(TM,'%.6d');
        imPath = ['S:\SiMView3\15-06-11\Dme_E2_His2AvRFP_spiderGFP_12-03_20150611_155054.corrected\SPM00\TM' TMstr '\']
        
        if( isMembrane == 0 )
            %for nuclear channel
            imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN01.klb'], ['SPM00_TM' TMstr '_CM02_CHN02.klb'], ['SPM00_TM' TMstr '_CM01_CHN01.klb'], ['SPM00_TM' TMstr '_CM03_CHN02.klb']};%0,90,180,270 degrees
            outputFolder = ['\\dm11\kellerlab\deconvolutionDatasets\15_06_11_fly_functionalImage_TM' TMstr '_Fiji\']
        else
            %for membrane channel
            imFilenameCell = {['SPM00_TM' TMstr '_CM00_CHN03.klb'], ['SPM00_TM' TMstr '_CM02_CHN00.klb'], ['SPM00_TM' TMstr '_CM01_CHN03.klb'], ['SPM00_TM' TMstr '_CM03_CHN00.klb']};%0,90,180,270 degrees
            outputFolder = ['T:\temp\deconvolution\15_06_11_fly_functionalImage_TM' TMstr 'membrCh_Fiji\']
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
            
            %flip and permutation and interpolation in z
            %tform_ = coarseRegistrationBasedOnMicGeometry(im,angles(ii), 1, imRefSize);%FIji already has the anisotropy incorporated
            
            %A = tform_
            
            %sprintf('%.1f, ', A(:,1:3))
            %        addpath './imWarpFast/'
            %        imTemp = imwarpfast(im, tform_, 2, imRefSize);
            %        rmpath './imWarpFast/'
            
            %write image
            writeTifStack(im,[outputFolder 'simview3_view' num2str(ii-1)]);
            %       writeRawStack(im,[outputFolder 'simview3_view' num2str(ii-1) '.raw']);
        end
    end
end