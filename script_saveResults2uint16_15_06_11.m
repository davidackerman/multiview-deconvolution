function script_saveResults2uint16_15_06_11(TM)

folder = {'','membrCh'};
suffix = {'imReg_', 'weightsReg_'};
scale = [100, 100];%even if we could scale further, it affects compressibility

numViews = 4;

outputFolder = ['S:\temp\deconvolution\15_06_11_dualChannel\regData\'];

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

if(exist('b:') == 0)
   return; 
end



%%
for ff = 1:length(folder)
    imgPattern = [networkFolder '\TM??????' folder{ff} '_Fiji\'];        
   
    out = [outputFolder 'TM' num2str(TM,'%.6d') folder{ff} '_Fiji\'];
    
    if( exist(out,'dir') == 0 )
        mkdir(out);
    end
    
    for ss = 1:length(suffix)
        basename = [recoverFilenameFromPattern(imgPattern,TM) suffix{ss}];
        for ii = 1:numViews
            filename = [basename num2str(ii) '.klb'];
            if( exist(filename,'file') == 0)
                continue;
            end
            im = readKLBstack(filename);            
            im = uint16(scale(ss) * im );
            writeKLBstack(im, [out suffix{ss} num2str(ii) '_uint16_sc' num2str(scale(ss)) '.klb'] );
        end
    end
        
    name = 'dataset.xml';
    copyfile([recoverFilenameFromPattern(imgPattern,TM) name],[out name]);
    name = 'Macro.ijm';
    copyfile([recoverFilenameFromPattern(imgPattern,TM) name],[out name]);
end