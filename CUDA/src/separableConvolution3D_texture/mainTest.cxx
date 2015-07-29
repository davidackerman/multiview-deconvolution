/*
* Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation and 
* any modifications thereto.  Any use, reproduction, disclosure, or distribution 
* of this software and related documentation without an express license 
* agreement from NVIDIA Corporation is strictly prohibited.
* 
*/


/* 
* This sample implements the same algorithm as the convolutionSeparable
* CUDA SDK sample, but without using the shared memory at all.
* Instead, it uses textures in exactly the same way an OpenGL-based
* implementation would do. 
* Refer to the "Performance" section of convolutionSeparable whitepaper.
*/

//updgraded to do 3d convolutions.



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include "convolutionTexture_common.h"

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    float
        *h_Input,
        *h_Buffer,
        *h_Buffer2,
        *h_OutputCPU,
        *h_OutputGPU,
        *pulseAx  = 0, 
        *pulseLat = 0, 
        *pulseEle = 0,
        *d_Output;



	int iterations = 10;
	int imageW = 63;
    int imageH = 90;
    int imageD = 34; 
	int kernel_length_ax = 5;
	int kernel_length_lat = 11;
	int kernel_length_ele = 7;
    

	if( argc > 1 )
		iterations = atoi( argv[1] );
	if( argc > 2 )
	{
		imageW = atoi(argv[2]);
		imageH = atoi(argv[3]);
		imageD = atoi(argv[4]);
	} 

	if( argc > 5 )
	{
		kernel_length_lat = atoi(argv[5]);
		kernel_length_ax = atoi(argv[6]);
		kernel_length_ele = atoi(argv[7]);
	} 

    cudaArray
        *a_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();      

    float
        gpuTime    = 0,
        gpuTimeTot = 0;

    StopWatchInterface* hTimer;

    const int numPixels = imageW * imageH * imageD;
	const float sigma[3] ={ kernel_length_ax / 3.0f, kernel_length_lat / 3.0f, kernel_length_ele / 3.0f};
    int lAx, lLat, lEle;

	
	//---------------------------------------------------------------
	//generate kernels
	lAx = kernel_length_ax;
	pulseAx = (float*)malloc( sizeof(float) * lAx );
	for(int ii = 0; ii < lAx; ii++)
		pulseAx[ii] = exp( pow( float(ii- (lAx-1)/2 ) / sigma[0], 2.0f) ) / sqrt( 2.0f * 3.14159f * sigma[0] * sigma[0]);

	lLat = kernel_length_lat;
	pulseLat = (float*)malloc( sizeof(float) * lLat );
	for(int ii = 0; ii < lLat; ii++)
		pulseLat[ii] = exp( pow( float(ii- (lLat-1)/2 ) / sigma[1], 2.0f) ) / sqrt( 2.0f * 3.14159f * sigma[1] * sigma[1]);
	
	lEle = kernel_length_ele;
	pulseEle = (float*)malloc( sizeof(float) * lEle );
	for(int ii = 0; ii < lEle; ii++)
		pulseEle[ii] = exp( pow( float(ii- (lEle-1)/2 ) / sigma[2], 2.0f) ) / sqrt( 2.0f * 3.14159f * sigma[2] * sigma[2]);

    //--------------------------------------------------------------    
    
    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("Size is %d x %d x %d = %d voxels\n", imageW, imageH, imageD, numPixels);
    h_Input        = (float *)malloc(numPixels * sizeof(float));
    h_Buffer       = (float *)malloc(numPixels * sizeof(float));
    h_Buffer2      = (float *)malloc(numPixels * sizeof(float));
    h_OutputCPU    = (float *)malloc(numPixels * sizeof(float));
    h_OutputGPU    = (float *)malloc(numPixels * sizeof(float));


    cudaExtent volExtent = make_cudaExtent( imageW, imageH, imageD);
    checkCudaErrors( cudaMalloc3DArray(&a_Src, &floatTex, volExtent) );
    checkCudaErrors( cudaMalloc((void **)&d_Output, numPixels * sizeof(float)) );
    printf("allocated memory space\n");
    
    //create a random 3D field
    srand(2009);
    for(int i = 0; i < imageW * imageH * imageD; i++)
        h_Input[i] = (float)(rand());// % 16);
        
  
    setConvolutionKernelAx (pulseAx, kernel_length_ax);
    setConvolutionKernelLat(pulseLat, kernel_length_lat);
    setConvolutionKernelEle(pulseEle, kernel_length_ele);


    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*) (h_Input), volExtent.width*sizeof(float), volExtent.width, volExtent.height);
    copyParams.dstArray = a_Src;
    copyParams.extent   = volExtent;
    copyParams.kind     = cudaMemcpyHostToDevice;

    checkCudaErrors( cudaMemcpy3D(&copyParams) );

	//------debugging: check result---------------------
	printf("==========DEBUGGING: writing result to E:/temp/testWatershed/imRandomInput.bin in float32 with size %d x %d x %d\n", imageW, imageH, imageD);;		
	FILE* fid = fopen("E:/temp/testWatershed/imRandomInput.bin", "wb");	
	fwrite(h_Input,sizeof(float), numPixels, fid);	
	fclose(fid);

	fid = fopen("E:/temp/testWatershed/imRandomKernel_x.bin", "wb");	
	fwrite(pulseAx,sizeof(float), kernel_length_ax, fid);	
	fclose(fid);

	fid = fopen("E:/temp/testWatershed/imRandomKernel_y.bin", "wb");	
	fwrite(pulseLat,sizeof(float), kernel_length_lat, fid);	
	fclose(fid);

	fid = fopen("E:/temp/testWatershed/imRandomKernel_z.bin", "wb");	
	fwrite(pulseEle,sizeof(float), kernel_length_ele, fid);	
	fclose(fid);
	//---------------------------------------------------
   

    ////////////////////////////////////////////////////////////////////
    //Doing the row convolution
    printf("Running GPU rows convolution (%u identical iterations)...\n", iterations);    	
    for(int i = 0; i < iterations; i++){

		//i == -1 -- warmup iteration
		if (i == 0)
		{
			checkCudaErrors( cudaDeviceSynchronize() );
			sdkResetTimer(&hTimer) ;
			sdkStartTimer(&hTimer) ;
		}
        convolutionRowsGPUtexture(
            d_Output,
            a_Src,
            imageW,
            imageH,
            imageD,
			(kernel_length_lat - 1 ) / 2
            );
    }
    checkCudaErrors( cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer) ;
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    gpuTimeTot += gpuTime;
    printf("Average convolutionRowsGPU() time: %f msecs; //%f Mpix/s for kernel size = %d pixels\n", gpuTime, imageW * imageH * imageD * 1e-6 / (0.001 * gpuTime), lLat);
 
    //While CUDA kernels can't write to textures directly, this copy is inevitable
    printf("Copying convolutionRowGPU() output back to the texture...\n");
    checkCudaErrors( cudaDeviceSynchronize() );
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    copyParams.srcPtr   = make_cudaPitchedPtr((void*) (d_Output), volExtent.width*sizeof(float), volExtent.width, volExtent.height);
    copyParams.dstArray = a_Src; 
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D(&copyParams) ); 

    checkCudaErrors( cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    gpuTimeTot += gpuTime;
    printf("cudaMemcpy3D(deviceToDevice) time: %f msecs; //%f Mpix/s\n\n", gpuTime, imageW * imageH * imageD * 1e-6 / (0.001 * gpuTime));


    /////////////////////////////////////////////////////
    //doing the column convolution

    printf("Running GPU columns convolution (%i iterations)\n", iterations);
    for(int i = 0; i < iterations; i++){

		//i == -1 -- warmup iteration
		if (i == 0)
		{
			checkCudaErrors( cudaDeviceSynchronize() );
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}
		convolutionColumnsGPUtexture(d_Output, a_Src, imageW, imageH, imageD, (kernel_length_ax - 1 ) / 2);
    }
    checkCudaErrors( cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    gpuTimeTot += gpuTime;
    printf("Average convolutionColumnsGPU() time: %f msecs; //%f Mpix/s for kernel size = %d pixels\n", gpuTime, imageW * imageH * imageD * 1e-6 / (0.001 * gpuTime), lAx);
    printf("Reading back GPU results...\n");

    checkCudaErrors(   cudaDeviceSynchronize() );
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    copyParams.srcPtr   = make_cudaPitchedPtr((void*) (d_Output), volExtent.width*sizeof(float), volExtent.width, volExtent.height);
    copyParams.dstArray = a_Src; 
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D(&copyParams) );

    checkCudaErrors(   cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    gpuTimeTot += gpuTime;
    printf("cudaMemcpy(deviceTodevice) time: %f msecs; //%f Mpix/s\n\n", gpuTime, imageW * imageH * imageD * 1e-6 / (0.001 * gpuTime));

    /////////////////////////////////////////////
    //doing the depth convolution
    printf("Running GPU depth convolution (%i iterations)\n", iterations);    

    for(int i = 0; i < iterations; i++){
		//i == -1 -- warmup iteration
		if (i == 0)
		{
			checkCudaErrors( cudaDeviceSynchronize() );
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}
        convolutionDepthGPUtexture(
            d_Output,
            a_Src,
            imageW,
            imageH,
            imageD,
			(kernel_length_ele - 1 ) / 2
            );
    }
    checkCudaErrors( cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    gpuTimeTot += gpuTime;
    printf("Average convolutionDepthGPU() time: %f msecs; //%f Mpix/s for kernel size = %d pixels\n", gpuTime, imageW * imageH * imageD * 1e-6 / (0.001 * gpuTime), lEle);


    //copying from device back to host
    checkCudaErrors   ( cudaDeviceSynchronize() );
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    checkCudaErrors   ( cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH *imageD * sizeof(float), cudaMemcpyDeviceToHost) );
    checkCudaErrors   ( cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    gpuTimeTot += gpuTime;
    printf("cudaMemcpy(deviceToHost) time: %f msecs; //%f Mpix/s\n\n", gpuTime, imageW * imageH * imageD * 1e-6 / (0.001 * gpuTime));
    
	printf("GPU total time (including memory transfers) (msec): %f; //%f Mpix/s \n\n", gpuTimeTot, imageW * imageH * imageD * 1e-6 / (0.001 * gpuTimeTot));
    
    
    //now compare to the CPU results for consistancy
    //////////////////////////////////////

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    printf("Checking the results...\n");
    printf("...running convolutionRowsCPU()");
    convolutionRowsCPU(
        h_Buffer,
        h_Input,
        pulseLat,
        imageW,
        imageH,
        imageD,
        (kernel_length_lat - 1 ) / 2
        );

    printf("...done\n...running convolutionColumnsCPU()");
    convolutionColumnsCPU(
        h_Buffer2,
        h_Buffer,
        pulseAx,
        imageW,
        imageH,
        imageD,
        (kernel_length_ax - 1 ) / 2
        );

    printf("...done\n...running convolutionDepthCPU()");
    convolutionDepthCPU(
        h_OutputCPU,
        h_Buffer2,
        pulseEle,
        imageW,
        imageH,
        imageD,
        (kernel_length_ele - 1 ) / 2
        );
    printf("...done\n\n");
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("CPU total time: %f msecs \n", gpuTime);
    printf("speedup: %f\n\n", gpuTime/gpuTimeTot);



    double delta = 0; 
    double sum = 0;
    double sad = 0; //sum absolute difference
    for(int i = 0; i < imageW * imageH * imageD; i++)
	{
        sum   += h_OutputCPU[i] * h_OutputCPU[i];
        delta +=(h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
        sad   += abs(h_OutputCPU[i] - h_OutputGPU[i]);
    }
    double L2norm = sqrt(delta / sum);
    printf("Relative L2 norm: %E\n", L2norm);
    printf("SAD %E \n", sad);
    printf((L2norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n"); 


	printf("==========DEBUGGING: writing result to E:/temp/testWatershed/ in float32 with size %d x %d x %d\n", imageW, imageH, imageD);;		
	fid = fopen("E:/temp/testWatershed/h_OutputCPU.bin", "wb");	
	fwrite(h_OutputCPU,sizeof(float), numPixels, fid);	
	fclose(fid);

	fid = fopen("E:/temp/testWatershed/h_OutputGPU.bin", "wb");	
	fwrite(h_OutputGPU,sizeof(float), numPixels, fid);	
	fclose(fid);

   // view the images if you have cimg installed or don't bother. 
   // frameImage(h_Input, imageH, imageW, "h_Input");
   // frameImage(h_OutputGPU, imageW, imageH, "h_OutputGpu");
   // frameImage(h_OutputCPU, imageW, imageH, "h_OutputCPU");


    printf("Shutting down...\n");
    checkCudaErrors( cudaFree(d_Output)   );
    checkCudaErrors( cudaFreeArray(a_Src)   );

    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Buffer2);
    free(h_Input);

    free(pulseAx);
    free(pulseLat);
    free(pulseEle);


    //cutilExit(argc, argv);

    cudaThreadExit();
}
