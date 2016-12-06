function result = max_conditioned_relative_error(output_stack_cpp, output_stack_matlab, scale)

    if isequal(size(output_stack_cpp), size(output_stack_matlab)) ,
        rel_err = abs(output_stack_cpp-output_stack_matlab)./max(scale,abs(output_stack_matlab)) ;
        result = max(max(max(rel_err))) ;
    else
        result = inf ;
    end

end
