classdef three_identity_views_test_case < matlab.unittest.TestCase
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
        function the_test(self)
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
    end  % test methods

end  % classdef
