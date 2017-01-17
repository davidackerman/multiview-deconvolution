classdef simple_transforms_test_case < matlab.unittest.TestCase
    properties 
        old_path
    end
    
    methods (TestMethodSetup)
        function setup(self) 
            % Add KLB reader, writer to path
            self.old_path = modpath('release') ;
        end
    end

    methods (TestMethodTeardown)
        function teardown(self) 
            path(self.old_path) ;
        end
    end

    methods (Test)
        function test_with_identity_transforms(self)
            % Make a PSF with a isotropic gaussian in it
            spacing = [1 1 1] ;  % um
            FWHM = [2 2 2] ;  % um
            n_sigmas = 8 ;
            psf = generate_PSF(spacing, FWHM, n_sigmas) ;  % elements sum to unity
            psf = trim_PSF(psf, 1e-10) ;
            %psf_dims = size(psf)

            % Make a stack with a little cube in the middle
            n_cube_size = 10 ;  % n voxels on a side
            n = 128 ;
            n_x = n ;
            n_y = n ;
            n_z = n ;
            truth = zeros([n_y n_x n_z]) ;
            truth(1+n/2-n_cube_size/2:1+n/2+n_cube_size/2, ...
                  1+n/2-n_cube_size/2:1+n/2+n_cube_size/2, ...
                  1+n/2-n_cube_size/2:1+n/2+n_cube_size/2) = ...
                30000 ;
            %writeKLBstack(single(truth),'truth.klb') ;
            %figure('color','w'); imagesc(truth(:,:,64),[0 max3(truth)]); axis square; title('truth'); colorbar

            % Convolve it (once) with the isotropic PSF
            %tic
            input = convn(truth, psf, 'same') ;
            %toc
            %writeKLBstack(single(blurred_stack),'blurred_stack_as_float32.klb') ;
            %figure('color','w'); imagesc(input(:,:,64),[0 max3(input)]); axis square; title('input'); colorbar

            % The transform matrix
            A = [ 1 0  0   0 ; ...
                  0 1  0   0 ; ...
                  0 0  1   0 ; ...
                  0 0  0   1 ] ;

            % Output each of the views, psfs to a .klb, so we can use them for testing.
            psf_file_name = horzcat(tempname(), '.klb') ;
            writeKLBstack(single(psf),psf_file_name) ;
            input_file_name = horzcat(tempname(), '.klb') ;            
            writeKLBstack(single(input),input_file_name) ;

            % Output a .smv.xml to give as input to the MVD code
            smv_xml_file_name = horzcat(tempname(), '.smv.xml') ;         
            input_file_names = {input_file_name input_file_name input_file_name} ;
            psf_file_names = {psf_file_name psf_file_name psf_file_name} ;
            As = { A A A } ;  % Put the identity-like transform as first view
            deconvolution_parameters = struct('blockZsize', {-1}, ...
                                              'imBackground', {0}, ...
                                              'lambdaTV', {0}, ...
                                              'numIter', {15}, ...
                                              'verbose', {0}, ...
                                              'saveAsUINT16', {0}) ;
            save_smv_xml(smv_xml_file_name, input_file_names, psf_file_names, As, deconvolution_parameters) ;

            % Determine absolute path to the .exe
            absolute_path_to_this_file = mfilename('fullpath') ;
            absolute_path_to_this_dir = fileparts(absolute_path_to_this_file) ;
            absolute_path_to_mvd_repo = fileparts(absolute_path_to_this_dir) ;
            absolute_path_to_exe_dir = fullfile(absolute_path_to_mvd_repo, 'CUDA/build/src/Release') ;
            absolute_path_to_exe = fullfile(absolute_path_to_exe_dir, 'multiview_deconvolution_LR_multiGPU.exe') ;

            % Run the code the old way, with PSF-transforming done externally
            fprintf('\n\nRunning executable...') ;
            n_gpus = -1 ;  % means use all
            output_file_name = horzcat(tempname(), '.klb') ;
            command_line = ...
                sprintf('%s %s %d %s', ...
                        absolute_path_to_exe, ...
                        smv_xml_file_name, ...
                        n_gpus, ...
                        output_file_name) ;
            fprintf('%s\n', command_line) ;        
            status = system(command_line) ;
            if status==0 ,
                fprintf('Run exited normally.\n') ;
            else
                error('.exe returned exit code %d', status) ;
            end

            % Read in the output stack and examine it
            output = readKLBstack(output_file_name) ;
            self.verifyEqual(class(output), 'single') ;

            % Make sure output is not all-zero...
            is_output_nonzero = any(any(any(output))) ;
            self.verifyTrue(is_output_nonzero, 'The output stack is everywhere zero, which should not be.') ;
            %figure('color','w'); imagesc(output(:,:,64),[0 output_stack_max]); axis square; title('output'); colorbar

            % Compute the output error, make sure it's low enough
            mae_of_input = mean(mean(mean(abs(input-truth)))) ;
            fprintf('Mean absolute error of input (compared to the unblurred stack) is %g\n', mae_of_input) ;
            mae = mean(mean(mean(abs(output-truth)))) ;
            fprintf('Mean absolute error of output is %g\n', mae) ;
            mae_threshold = 7 ;  % After 15 iterations, error should be below this value (this was established empirically)
            self.verifyTrue(mae<mae_threshold, 'The output stack differs from the truth by too much') ;
            
            % Delete the temporary files (should also do this on error
            % exit...)
            delete(output_file_name) ;
            delete(smv_xml_file_name) ;
            delete(psf_file_name) ;
            delete(input_file_name) ;
        end  % function    

        function test_with_simple_rotations(self)
            % Make a PSF with a isotropic gaussian in it
            spacing = [1 1 1] ;  % um
            FWHM = [2 2 2] ;  % um
            n_sigmas = 8 ;
            psf = generate_PSF(spacing, FWHM, n_sigmas) ;  % elements sum to unity
            psf = trim_PSF(psf, 1e-10) ;
            %psf_dims = size(psf)

            % Make a stack with a little cube in the middle
            n_cube_size = 10 ;  % n voxels on a side
            n = 128 ;
            n_x = n ;
            n_y = n ;
            n_z = n ;
            truth = zeros([n_y n_x n_z]) ;
            truth(1+n/2-n_cube_size/2:1+n/2+n_cube_size/2, ...
                  1+n/2-n_cube_size/2:1+n/2+n_cube_size/2, ...
                  1+n/2-n_cube_size/2:1+n/2+n_cube_size/2) = ...
                30000 ;
            %writeKLBstack(single(truth),'truth.klb') ;
            %figure('color','w'); imagesc(truth(:,:,64),[0 max3(truth)]); axis square; title('truth'); colorbar

            % Convolve it (once) with the isotropic PSF
            %tic
            input = convn(truth, psf, 'same') ;
            %toc
            %writeKLBstack(single(blurred_stack),'blurred_stack_as_float32.klb') ;
            %figure('color','w'); imagesc(input(:,:,64),[0 max3(input)]); axis square; title('input'); colorbar

            % View 1 is looking along the z-axis, looking from z = -inf to z = +inf
            A_1 = [ 1 0  0   0 ; ...
                    0 1  0   0 ; ...
                    0 0  1   0 ; ...
                    0 0  0   1 ]' ;

            % View 2 is looking along the x-axis, looking from x = -inf to x = +inf
            A_2 = [ 0 0 -1  n+2 ; ...
                    0 1  0   0 ; ...
                    1 0  0   0 ; ...
                    0 0  0   1 ]' ;
            % That +2 is needed to get all the xtransformed stack to line up perfectly
            % (?!)
                
            % View 3 is looking along the y-axis, looking from y = -inf to y = +inf
            A_3 = [ 1 0  0   0 ; ...
                    0 0 -1  n+2 ; ...
                    0 1  0   0 ; ...
                    0 0  0   1 ]' ;
            % That +2 is needed to get all the xtransformed stack to line up perfectly
            % (?!)

            % Output each of the views, psfs to a .klb, so we can use them for testing.
            psf_file_name = horzcat(tempname(), '.klb') ;
            writeKLBstack(single(psf),psf_file_name) ;
            input_file_name = horzcat(tempname(), '.klb') ;            
            writeKLBstack(single(input),input_file_name) ;

            % Output a .smv.xml to give as input to the MVD code
            smv_xml_file_name = horzcat(tempname(), '.smv.xml') ;         
            input_file_names = {input_file_name input_file_name input_file_name} ;
            psf_file_names = {psf_file_name psf_file_name psf_file_name} ;
            As = { A_1 A_2 A_3 } ;  % Put the identity-like transform as first view
            deconvolution_parameters = struct('blockZsize', {-1}, ...
                                              'imBackground', {0}, ...
                                              'lambdaTV', {0}, ...
                                              'numIter', {15}, ...
                                              'verbose', {1}, ...
                                              'saveAsUINT16', {0}) ;
            save_smv_xml(smv_xml_file_name, input_file_names, psf_file_names, As, deconvolution_parameters) ;

            % Determine absolute path to the .exe
            absolute_path_to_this_file = mfilename('fullpath') ;
            absolute_path_to_this_dir = fileparts(absolute_path_to_this_file) ;
            absolute_path_to_mvd_repo = fileparts(absolute_path_to_this_dir) ;
            absolute_path_to_exe_dir = fullfile(absolute_path_to_mvd_repo, 'CUDA/build/src/Release') ;
            absolute_path_to_exe = fullfile(absolute_path_to_exe_dir, 'multiview_deconvolution_LR_multiGPU.exe') ;

            % Run the code the old way, with PSF-transforming done externally
            fprintf('\n\nRunning executable...') ;
            n_gpus = -1 ;  % means use all
            output_file_name = horzcat(tempname(), '.klb') ;
            command_line = ...
                sprintf('%s %s %d %s', ...
                        absolute_path_to_exe, ...
                        smv_xml_file_name, ...
                        n_gpus, ...
                        output_file_name) ;
            fprintf('%s\n', command_line) ;        
            status = system(command_line) ;
            if status==0 ,
                fprintf('Run exited normally.\n') ;
            else
                error('.exe returned exit code %d', status) ;
            end

            % Check the first view after transforming, which should be identical to the
            % input stack
            view_1_registered_file_name = sprintf('%s_debug_img_0.klb',smv_xml_file_name) ;
            view_1_registered_stack = readKLBstack(view_1_registered_file_name) ;
            self.verifyEqual(class(view_1_registered_stack), 'single') ;
            max_abs_error_view_1_registered = max(max(max(abs(view_1_registered_stack-input))))
            self.verifyLessThan(max_abs_error_view_1_registered, 0.01, 'Registered view 1 (of 3) does not match input') ;

            view_2_registered_file_name = sprintf('%s_debug_img_1.klb',smv_xml_file_name) ;
            view_2_registered_stack = readKLBstack(view_2_registered_file_name) ;
            self.verifyEqual(class(view_2_registered_stack), 'single') ;
            max_abs_error_view_2_registered = max(max(max(abs(view_2_registered_stack-input))))
            self.verifyLessThan(max_abs_error_view_2_registered, 0.01, 'Registered view 2 (of 3) does not match input') ;

            view_3_registered_file_name = sprintf('%s_debug_img_2.klb',smv_xml_file_name) ;
            view_3_registered_stack = readKLBstack(view_3_registered_file_name) ;
            self.verifyEqual(class(view_3_registered_stack), 'single') ;
            max_abs_error_view_3_registered = max(max(max(abs(view_3_registered_stack-input))))            
            self.verifyLessThan(max_abs_error_view_3_registered, 0.01, 'Registered view 3 (of 3) does not match input') ;
            
            % Read in the output stack and examine it
            output = readKLBstack(output_file_name) ;
            self.verifyEqual(class(output), 'single') ;

            % Make sure output is not all-zero...
            is_output_nonzero = any(any(any(output))) ;
            self.verifyTrue(is_output_nonzero, 'The output stack is everywhere zero, which should not be.') ;
            %figure('color','w'); imagesc(output(:,:,64),[0 output_stack_max]); axis square; title('output'); colorbar

            % Compute the output error, make sure it's low enough
            mae_of_input = mean(mean(mean(abs(input-truth)))) ;
            fprintf('Mean absolute error of input (compared to the unblurred stack) is %g\n', mae_of_input) ;
            mae = mean(mean(mean(abs(output-truth)))) ;
            fprintf('Mean absolute error of output is %g\n', mae) ;
            mae_threshold = 7 ;  % After 15 iterations, error should be below this value (this was established empirically)
            self.verifyTrue(mae<mae_threshold, 'The output stack differs from the truth by too much') ;
            
            % Delete the temporary files (should also do this on error
            % exit...)
            delete(output_file_name) ;
            delete(view_3_registered_file_name) ;
            delete(view_2_registered_file_name) ;
            delete(view_1_registered_file_name) ;
            delete(smv_xml_file_name) ;
            delete(psf_file_name) ;
            delete(input_file_name) ;
        end  % function    
        
        function test_psf_transform_within_mvd(self)
            % Make a PSF with a isotropic gaussian in it
            spacing = [1 1 1] ;  % um
            FWHM = [2 2 2] ;  % um
            n_sigmas = 8 ;
            psf = generate_PSF(spacing, FWHM, n_sigmas) ;  % elements sum to unity
            psf = trim_PSF(psf, 1e-10) ;
            %psf_dims = size(psf)

            % Make a stack with a little cube in the middle
            n_cube_size = 10 ;  % n voxels on a side
            n = 128 ;
            n_x = n ;
            n_y = n ;
            n_z = n ;
            truth = zeros([n_y n_x n_z]) ;
            truth(1+n/2-n_cube_size/2:1+n/2+n_cube_size/2, ...
                  1+n/2-n_cube_size/2:1+n/2+n_cube_size/2, ...
                  1+n/2-n_cube_size/2:1+n/2+n_cube_size/2) = ...
                30000 ;
            %writeKLBstack(single(truth),'truth.klb') ;
            %figure('color','w'); imagesc(truth(:,:,64),[0 max3(truth)]); axis square; title('truth'); colorbar

            % Convolve it (once) with the isotropic PSF
            %tic
            input = convn(truth, psf, 'same') ;
            %toc
            %writeKLBstack(single(blurred_stack),'blurred_stack_as_float32.klb') ;
            %figure('color','w'); imagesc(input(:,:,64),[0 max3(input)]); axis square; title('input'); colorbar

            % View 1 is looking along the z-axis, looking from z = -inf to z = +inf
            A_1 = [ 1 0  0   0 ; ...
                    0 1  0   0 ; ...
                    0 0  1   0 ; ...
                    0 0  0   1 ]' ;

            % View 2 is looking along the x-axis, looking from x = -inf to x = +inf
            A_2 = [ 0 0 -1  n+2 ; ...
                    0 1  0   0 ; ...
                    1 0  0   0 ; ...
                    0 0  0   1 ]' ;
            % That +2 is needed to get all the xtransformed stack to line up perfectly
            % (?!)
                
            % View 3 is looking along the y-axis, looking from y = -inf to y = +inf
            A_3 = [ 1 0  0   0 ; ...
                    0 0 -1  n+2 ; ...
                    0 1  0   0 ; ...
                    0 0  0   1 ]' ;
            % That +2 is needed to get all the xtransformed stack to line up perfectly
            % (?!)

            % Output each of the views, psfs to a .klb, so we can use them for testing.
            psf_file_name = horzcat(tempname(), '.klb') ;
            writeKLBstack(single(psf),psf_file_name) ;
            input_file_name = horzcat(tempname(), '.klb') ;            
            writeKLBstack(single(input),input_file_name) ;

            % Determine absolute path to the .exe
            absolute_path_to_this_file = mfilename('fullpath') ;
            absolute_path_to_this_dir = fileparts(absolute_path_to_this_file) ;
            absolute_path_to_mvd_repo = fileparts(absolute_path_to_this_dir) ;
            absolute_path_to_exe_dir = fullfile(absolute_path_to_mvd_repo, 'CUDA/build/src/Release') ;
            absolute_path_to_exe = fullfile(absolute_path_to_exe_dir, 'multiview_deconvolution_LR_multiGPU.exe') ;

            % Output a .smv.xml to give as input to the MVD code
            smv_xml_file_name = horzcat(tempname(), '.smv.xml') ;         
            input_file_names = {input_file_name input_file_name input_file_name} ;
            psf_file_names = {psf_file_name psf_file_name psf_file_name} ;
            As = { A_1 A_2 A_3 } ;  % Put the identity-like transform as first view
            deconvolution_parameters = struct('blockZsize', {-1}, ...
                                              'imBackground', {0}, ...
                                              'lambdaTV', {0}, ...
                                              'numIter', {1}, ...
                                              'verbose', {0}, ...
                                              'saveAsUINT16', {0}) ;
            save_smv_xml(smv_xml_file_name, input_file_names, psf_file_names, As, deconvolution_parameters) ;

            % Run the code the old way, with PSF-transforming done externally
            fprintf('\n\nRunning executable...') ;
            n_gpus = -1 ;  % means use all
            output_file_name = horzcat(tempname(), '.klb') ;
            command_line = ...
                sprintf('%s %s %d %s', ...
                        absolute_path_to_exe, ...
                        smv_xml_file_name, ...
                        n_gpus, ...
                        output_file_name) ;
            fprintf('%s\n', command_line) ;        
            status = system(command_line) ;
            if status==0 ,
                fprintf('Run exited normally.\n') ;
            else
                error('.exe returned exit code %d', status) ;
            end
            
            % Read in the output stack and examine it
            output_with_external_psf_transform = readKLBstack(output_file_name) ;
            self.verifyEqual(class(output_with_external_psf_transform), 'single') ;

            % Make sure output is not all-zero...
            is_output_nonzero = any(any(any(output_with_external_psf_transform))) ;
            self.verifyTrue(is_output_nonzero, 'The output stack is everywhere zero, which should not be.') ;
            %figure('color','w'); imagesc(output(:,:,64),[0 output_stack_max]); axis square; title('output'); colorbar

            % Output a .smv.xml to give as input to the MVD code
            smv_xml_file_name_2 = horzcat(tempname(), '.smv.xml') ;         
            input_file_names = {input_file_name input_file_name input_file_name} ;
            psf_file_names = {psf_file_name psf_file_name psf_file_name} ;
            As = { A_1 A_2 A_3 } ;  % Put the identity-like transform as first view
            deconvolution_parameters = struct('blockZsize', {-1}, ...
                                              'imBackground', {0}, ...
                                              'lambdaTV', {0}, ...
                                              'numIter', {1}, ...
                                              'verbose', {0}, ...
                                              'saveAsUINT16', {0}, ...
                                              'isPSFAlreadyTransformed', {0}) ;
            save_smv_xml(smv_xml_file_name_2, input_file_names, psf_file_names, As, deconvolution_parameters) ;

            % Run the code the old way, with PSF-transforming done externally
            fprintf('\n\nRunning executable...') ;
            n_gpus = -1 ;  % means use all
            output_file_name_2 = horzcat(tempname(), '.klb') ;
            command_line = ...
                sprintf('%s %s %d %s', ...
                        absolute_path_to_exe, ...
                        smv_xml_file_name_2, ...
                        n_gpus, ...
                        output_file_name_2) ;
            fprintf('%s\n', command_line) ;        
            status = system(command_line) ;
            if status==0 ,
                fprintf('Run exited normally.\n') ;
            else
                error('.exe returned exit code %d', status) ;
            end
            
            % Read in the output stack and examine it
            output_with_internal_psf_transform = readKLBstack(output_file_name_2) ;
            self.verifyEqual(class(output_with_internal_psf_transform), 'single') ;
            
            % Compare the two outputs, should be nearly identical
            max_abs_diff = max(max(max(abs(output_with_internal_psf_transform-output_with_external_psf_transform)))) ;
            fprintf('Max absolute difference is %g\n', max_abs_diff) ;
            max_abs_diff_threshold = 0.01 ;  
            self.verifyTrue(max_abs_diff<max_abs_diff_threshold, 'The output stacks differ by too much') ;
            
            % Delete the temporary files (should also do this on error
            % exit...)
            delete(output_file_name) ;
            delete(output_file_name_2) ;
            delete(smv_xml_file_name) ;
            delete(smv_xml_file_name_2) ;
            delete(psf_file_name) ;
            delete(input_file_name) ;            
        end        

    end  % test methods

end  % classdef
