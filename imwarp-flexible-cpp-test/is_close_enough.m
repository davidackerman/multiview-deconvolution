function result = is_close_enough(output_stack_cpp, output_stack_matlab)

    if isequal(size(output_stack_cpp), size(output_stack_matlab)) ,
        mcre = max_conditioned_relative_error(output_stack_cpp, output_stack_matlab, 1e-4)
        result = (mcre<0.1) ;        
    else
        result = false ;
    end

end
