function writeRawStack(im,filename)

fid = fopen(filename,'wb');
fwrite(fid,im(:),class(im));
fclose(fid);

fid = fopen([filename '.txt'],'w');

fprintf(fid,'%d ',size(im));
fprintf(fid,'\n%s\n',class(im));
fclose(fid);