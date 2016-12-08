function x = read_as_raw(file_name, m, n, type_name)
   fid = fopen(file_name, 'rb') ;
   x = fread(fid,[m n], type_name) ;
   fclose(fid) ;
end
