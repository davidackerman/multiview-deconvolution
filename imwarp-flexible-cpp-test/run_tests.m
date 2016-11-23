spacing = [1 1 1] ;
FWHM = [2.5 5 10] ;
n_sigmas = 8 ;
input_stack = generate_PSF_alternate(spacing, FWHM, n_sigmas) ;
input_stack_dims = size(input_stack)

% scale the input stack, convert to uint16
input_stack = single(input_stack) ;

% Define the source frame, which is centered on the origin
% frame = ...
%     imref3d(size(input_stack), ...
%             0.5*size(input_stack,2)*[-1 +1], ...
%             0.5*size(input_stack,1)*[-1 +1], ...
%             0.5*size(input_stack,3)*[-1 +1])
origin = [0.5 0.5 0.5]' ;
spacing = ones(3,1) ;
frame = ...
    imref3d(size(input_stack), ...
            origin(2)+spacing(2)*[0 size(input_stack,2)], ...
            origin(1)+spacing(1)*[0 size(input_stack,1)], ...
            origin(3)+spacing(3)*[0 size(input_stack,3)])  

% Test 1
A= [ 1 0 0 0 ;
     0 1 0 0 ;
     0 0 1 0 ;
     0 0 0 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, origin, spacing, A, origin, spacing, input_stack_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, frame, affine3d(A), 'OutputView', frame) ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 2
A= [ 0 1 0 0 ;
     0 0 1 0 ;
     1 0 0 0 ;
     0 0 0 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, origin, spacing, A, origin, spacing, input_stack_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, frame, affine3d(A), 'OutputView', frame) ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 3
A= [ 0 2 0 0 ;
     0 0 1 0 ;
     1 0 0 0 ;
     0 0 0 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, origin, spacing, A, origin, spacing, input_stack_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, frame, affine3d(A), 'OutputView', frame) ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 4
A= [ 0 2 0 0 ;
     0 0 1 0 ;
     1 0 0 0 ;
     1 2 3 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, origin, spacing, A, origin, spacing, input_stack_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, frame, affine3d(A), 'OutputView', frame) ;
%is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
if ~is_cpp_output_close_to_matlab , error('Deviation!  Deviation!') ; end

% Test 5
A= [ 1 2 0 0 ;
     0 0 1 0 ;
     1 0 0 0 ;
     1 2 3 1 ] ;
output_stack_cpp = run_cpp_imwarp_flexible(input_stack, origin, spacing, A, origin, spacing, input_stack_dims) ;
output_stack_cpp_size = size(output_stack_cpp) ;
output_stack_matlab = imwarp(input_stack, frame, affine3d(A), 'OutputView', frame) ;
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
    output_stack_cpp = run_cpp_imwarp_flexible(input_stack, origin, spacing, A, origin, spacing, input_stack_dims) ;
    output_stack_cpp_size = size(output_stack_cpp) ;
    output_stack_matlab = imwarp(input_stack, frame, affine3d(A), 'OutputView', frame) ;
    %is_matlab_output_close_to_input = is_close_enough(output_stack_matlab, input_stack) ;
    is_cpp_output_close_to_matlab = is_close_enough(output_stack_cpp, output_stack_matlab) 
    % These seem to always match, in that both are zero everywhere.
    are_all_elements_zero = ~any(any(any(output_stack_matlab)))
    if ~is_cpp_output_close_to_matlab ,
        error('Deviation!  Deviation!') ;
    end
end

% OK, it passes all of these!

