function im = readRawStack(filename)

fid = fopen([filename '.txt'],'r');
imSize = str2num(fgetl(fid));
format_ = fgetl(fid);
fclose(fid);

fid = fopen(filename,'rb');
im = reshape(fread(fid,prod(imSize),format_),imSize);
fclose(fid);

