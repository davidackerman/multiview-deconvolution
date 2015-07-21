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
 
 

#ifndef CONVOLUTIONTEXTURE_COMMON_H
#define CONVOLUTIONTEXTURE_COMMON_H


#include <stdint.h>
#include <stdlib.h>


////////////////////////////////////////////////////////////////////////////////
// Convolution kernel size (the only parameter inlined in the code)
////////////////////////////////////////////////////////////////////////////////
#define MAX_KERNEL_LENGTH  200


struct cudaArray; // forward declaration of cudaArray so we do not need to include cuda headers here

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int imageD,
    int kernelR
);

extern "C" void convolutionColumnsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int imageD,
    int kernelR
);

extern "C" void convolutionDepthCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int imageD,
    int kernelR
);


////////////////////////////////////////////////////////////////////////////////
// GPU texture-based convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setConvolutionKernelAx(float *h_Kernel, size_t kernel_length);
extern "C" void setConvolutionKernelLat(float *h_Kernel, size_t kernel_length);
extern "C" void setConvolutionKernelEle(float *h_Kernel, size_t kernel_length);

extern "C" void convolutionRowsGPUtexture(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    int imageD,
	int kernel_radius,
	float step_size = 1.0f
);

extern "C" void convolutionColumnsGPUtexture(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    int imageD,
	int kernel_radius,
	float step_size = 1.0f
);


extern "C" void convolutionDepthGPUtexture(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    int imageD,
	int kernel_radius,
	float step_size = 1.0f
);


extern "C" void convolutionRowsGPU(
    float *d_Dst,
    const float *a_Src,
    int imageW,
    int imageH,
    int imageD,
	int kernel_radius
);

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    const float *a_Src,
    int imageW,
    int imageH,
    int imageD,
	int kernel_radius
);


extern "C" void convolutionDepthGPU(
    float *d_Dst,
    const float *a_Src,
    int imageW,
    int imageH,
    int imageD,
	int kernel_radius
);


//to directly accumulate results in the outpout (for example, sum of partial derivatives)
//it adds the value of the convoution to d_Dst, so you better set it to all zeros first
extern "C" void convolutionRowsGPU_add(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	);

extern "C" void convolutionColumnsGPU_add(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	);


extern "C" void convolutionDepthGPU_add(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	);

//to directly calculate magnitude of gradient without re-accessing memory
//it adds the value of the convoution to d_Dst, so you better set it to all zeros first
extern "C" void convolutionRowsGPU_square(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	);

extern "C" void convolutionColumnsGPU_square(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	);


extern "C" void convolutionDepthGPU_square(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	);

////////////////////////////////////////////////////////////////////////////////
// GPU Hessian using separable convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void HessianWithGaussianDerivativesGPU_texture(
                                      const float *img_HOST,
                                      float *Hessian_CUDA,
									  int64_t imgDims[3],
									  float sigma,
                                      const float stepSize[3],
									  int kernel_radius
                                      );


extern "C" void HessianWithGaussianDerivativesGPU_AnisotropyZ(
                                      const float *img_HOST,
                                      float *Hessian_CUDA,
									  int64_t imgDims[3],
									  float sigma,
                                      const float step_size_z,
									  int kernel_radius
                                      );


////////////////////////////////////////////////////////////////////////////////
// GPU Total variation calculation
////////////////////////////////////////////////////////////////////////////////
extern "C" void TV_gradient_norm(
	const float *img_CUDA,
	float *TV_grad_norm_CUDA,
	int64_t imgDims[3],
	float sigma,
	int kernel_radius
	);

extern "C" void TV_divergence(
	const float *img_CUDA,
	const float *TV_grad_norm_CUDA,
	float* temp_CUDA,
	float *TV_out,
	int64_t imgDims[3],
	float sigma,
	int kernel_radius
	);

#endif
