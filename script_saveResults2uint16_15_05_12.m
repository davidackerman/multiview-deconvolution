function script_saveResults2uint16_15_05_12(TM)

suffix = {'imReg_', 'weightsReg_'}
scale = [50, 100];%even if we could scale further, it affects compressibility
imgPattern = 'T:\temp\deconvolution\15_05_12_fly_functionalImage_cluster\TM??????\simview3_TM????_';
numViews = 4;

%%
for ss = 1:length(suffix)
    basename = [recoverFilenameFromPattern(imgPattern,TM) suffix{ss}];
    for ii = 1:numViews
        filename = [basename num2str(ii) '.klb'];        
        if( exist(filename,'file') == 0)
            continue;
        end
        im = readKLBstack(filename);
        
        im = uint16(scale(ss) * im );
        writeKLBstack(im, [basename num2str(ii) '_uint16_sc' num2str(scale(ss)) '.klb'] );
        delete(filename);
    end
end
