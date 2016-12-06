% Make a simple stack
x(:,:,1) = [ 1 4 7 10 ;
             2 5 8 11 ;
             3 6 9 12 ] ;
x(:,:,2) = 12 + x(:,:,1) ;

writeKLBstack(single(x), 'input.klb') ;

A = eye(4) ;
save('affine_transform_matrix_in_y_equals_Ax_form_row_major.txt', 'A', '-ascii') ;

