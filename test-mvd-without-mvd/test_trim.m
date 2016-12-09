r = 3 ;
y_serial = (-r:0.1:+r)' ;
n=length(y_serial) ;
x = repmat(y_serial'                , [n 1 n]) ;
y = repmat(y_serial                 , [1 n n]) ;
z = repmat(reshape(y_serial,[1 1 n]), [n n 1]) ;

r2 = x.^2 + y.^2 + z.^2 ;

stack = exp(-r2) + normrnd(0,0.1,[n n n]) ;
threshold = 0.5 ;

[stack_trimmed, n_to_trim_plus_one] = trim_PSF(stack, threshold) ;
n_to_trim_plus_one
[stack_trimmed_maybe, n_to_trim_plus_one_maybe, n_to_trim_maybe] = trim_PSF_better(stack,threshold) ;
n_to_trim_plus_one_maybe
n_to_trim_maybe

if isequal(size(stack_trimmed),size(stack_trimmed_maybe)) && isequal(n_to_trim_maybe+1,n_to_trim_plus_one) ,
    fprintf('Test passed.\n') ;
else
    fprintf('Test failed.\n') ;
end
