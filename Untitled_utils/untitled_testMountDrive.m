
if(exist('b:') == 0)
    fid = fopen('HHMIcredentials.txt','r');
    username = fgetl(fid);
    password = fgetl(fid);
    fclose(fid);
    %mount networkd drive
    cmd = ['net use b: \\fxt.int.janelia.org\nobackup /user:' username ' ' password];
    [rr,ss] = system(cmd);
else
    rr = -1;
    ss = 'Folder existed';
end

fid = fopen('Y:\exchange\Fernando\debugClusterOutput.txt','w');
fprintf(fid,'rr = %d\n',rr);
fprintf(fid,'%s\n',ss);
fclose(fid);

%%
fid = fopen('b:\keller\testMountDrive.txt','w');
fprintf(fid,'I was here\n');
fclose(fid);