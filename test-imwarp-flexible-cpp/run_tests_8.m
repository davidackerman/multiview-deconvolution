input_spacing = [1 2 4]' ;  % xyz
FWHM = [2.5 5 10] ;  % xyz
input_dims = [69 69 69] ; % yxz
input_stack = generate_PSF_with_given_dims(input_spacing, FWHM, input_dims) ;

% scale the input stack, convert to uint16
input_stack = single(input_stack) ;

% Define the frame
[n_y_input, n_x_input, n_z_input] = size(input_stack) ;
input_origin = -0.5 * (input_spacing .* [n_x_input n_y_input n_z_input]')
%spacing = ones(3,1) ;
input_frame = ...
    imref3d(size(input_stack), ...
            input_origin(1)+input_spacing(1)*[0 n_x_input], ...
            input_origin(2)+input_spacing(2)*[0 n_y_input], ...
            input_origin(3)+input_spacing(3)*[0 n_z_input])  

% Test 1
A= [ 1 0 0 0 ;
     0 1 0 0 ;
     0 0 1 0 ;
     0 0 0 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, A, input_origin, input_spacing, input_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, input_frame, affine3d(A), 'OutputView', input_frame, 'Interp', 'cubic') ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 2
A= [ 0 1 0 0 ;
     0 0 1 0 ;
     1 0 0 0 ;
     0 0 0 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, A, input_origin, input_spacing, input_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, input_frame, affine3d(A), 'OutputView', input_frame, 'Interp', 'cubic') ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 2.5
A= [ 1 0 0 0 ;
     0 2 0 0 ;
     0 0 1 0 ;
     0 0 0 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, A, input_origin, input_spacing, input_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, input_frame, affine3d(A), 'OutputView', input_frame, 'Interp', 'cubic') ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 3
A= [ 0 2 0 0 ;
     0 0 1 0 ;
     1 0 0 0 ;
     0 0 0 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, A, input_origin, input_spacing, input_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, input_frame, affine3d(A), 'OutputView', input_frame, 'Interp', 'cubic') ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 4
A= [ 0 2 0 0 ;
     0 0 1 0 ;
     1 0 0 0 ;
     1 2 3 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, A, input_origin, input_spacing, input_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, input_frame, affine3d(A), 'OutputView', input_frame, 'Interp', 'cubic') ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 5
A= [ 1 2 0 0 ;
     0 0 1 0 ;
     1 0 0 0 ;
     1 2 3 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, A, input_origin, input_spacing, input_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, input_frame, affine3d(A), 'OutputView', input_frame, 'Interp', 'cubic') ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 6
for i=1:5
    A = unifrnd(-0.05,0.05,4,4) ;
    A(:,end) = [0 0 0 1]' ;
    A(end,:) = [0 0 0 1] ;
    A(1,1) = 1 ;
    A(2,2) = 1 ;
    A(2,2) = 1 ;    
    output_stack_cpp = run_cpp_imwarp_flexible(input_stack, input_origin, input_spacing, A, input_origin, input_spacing, input_dims) ;
    output_stack_cpp_size = size(output_stack_cpp) ;
    output_stack_matlab = imwarp(input_stack, input_frame, affine3d(A), 'OutputView', input_frame, 'Interp', 'cubic') ;
    %is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
    is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
    % These seem to always match, in that both are zero everywhere.
    are_all_elements_zero = ~any(any(any(output_stack_matlab)))
    if ~is_cpp_output_close_to_matlab ,
        error('Deviation!  Deviation!') ;
    end
end

% OK, it passes all of these!

