function result = is_close_enough(test_stack, reference_stack)

    if isequal(size(test_stack), size(reference_stack)) ,
        mcre = max_conditioned_relative_error(test_stack, reference_stack, 1e-4)
        result = (mcre<1e-3) ;        
    else
        fprintf('Stacks are not same size in is_close_enough()\n') ;
        size_test_stack = size(test_stack)
        size_reference_stack = size(reference_stack)
        result = false ;
    end

end
