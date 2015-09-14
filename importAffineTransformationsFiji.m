function tformFiji = importAffineTransformationsFiji(xmlFilename)

fid = fopen(xmlFilename,'r');

tformFiji = cell(1);
viewNold = -1;
while(1)
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    
    if( ~isempty(strfind(tline,'ViewRegistration timepoint=')) )
        pos = strfind(tline,'"');
        viewN = str2double(tline(pos(3)+1:pos(4)-1));
        
        if( viewN ~= viewNold )
           viewNold = viewN;
           numT = 1;
           tformFiji{viewN+1} = cell(1);
        end
            
    elseif( ~isempty(strfind(tline,'</affine>')) )
        pos1 = strfind(tline, '>');
        pos2 = strfind(tline, '<');
        tformFiji{viewN+1}{numT} = reshape(str2num(tline(pos1(1)+1:pos2(2)-1)),[4 3]);
        tformFiji{viewN+1}{numT} = [tformFiji{viewN+1}{numT}, [0;0;0;1]];
        numT = numT + 1;
    end
    
end

fclose(fid);