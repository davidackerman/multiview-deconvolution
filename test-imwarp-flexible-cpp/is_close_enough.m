function [result, are_same_size] = is_close_enough(output_stack_cpp, output_stack_matlab)

    if isequal(size(output_stack_cpp), size(output_stack_matlab)) ,
        are_same_size = true ;
        mcre = max_conditioned_relative_error(output_stack_cpp, output_stack_matlab, 1e-4)
        result = (mcre<0.001) ;        
    else
        are_same_size = false ;
        result = false ;
    end

end
