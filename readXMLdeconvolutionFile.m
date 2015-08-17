function [Acell, imgFilenameCell] = readXMLdeconvolutionFile(filenameXML)

fid = fopen(filenameXML, 'r');
Acell = cell(1,1);
imgFilenameCell = Acell;
count = 0;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    
    if( length(tline) >= 5 && strcmp(tline(1:5),'<view') == 1)
        pp = findstr(tline,'A="') + 2;
        pp2 = findstr(tline,'"');
        
        count = count + 1;
        idx = find(pp2 == pp);
        Acell{count} = reshape(str2num(tline(pp+1:pp2(idx+1)-1)), [4 4]);
        
        p1 = findstr(tline,'imgFilename="');
        aux = tline(p1 + 13:end);
        p2 = findstr(aux,'"');
        imgFilenameCell{count} = aux(1:p2(1)-1);
    end
    
end

fclose(fid);