function write_as_raw(file_name, x)
   fid = fopen(file_name, 'wb') ;
   fwrite(fid,x,class(x)) ;
   fclose(fid) ;
end