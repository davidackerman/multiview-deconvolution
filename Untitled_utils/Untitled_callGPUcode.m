for iter = 30
    system(['C:\Users\Fernando\matlabProjects\deconvolution\CUDA\build\test\Release\test_multiview_deconvolution_LR_multiGPU.exe //dm11/kellerlab/deconvolutionDatasets/20150505_185415_GCaMP6_TM000089/simview3_TM89_ ' num2str(iter)]);
    system(['C:\Users\Fernando\matlabProjects\deconvolution\CUDA\build\test\Release\test_multiview_deconvolution_LR_multiGPU.exe //dm11/kellerlab/deconvolutionDatasets/15_05_12_fly_functionalImage_cluster/TM002971/simview3_TM2971_ ' num2str(iter)]);
end

