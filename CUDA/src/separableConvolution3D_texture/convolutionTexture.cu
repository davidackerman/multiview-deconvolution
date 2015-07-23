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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>
#include "convolutionTexture_common.h"
#include "book.h"



////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b){
	return (a % b != 0) ? (a - a % b + b) : a;
}


////////////////////////////////////////////////////////////////////////////////
// Auxiliary kernels
////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void elementwiseOperationInPlace_kernel_sqRoot(T *A, uint64_t arrayLength)
{
	uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (tid < arrayLength)
	{
		A[tid] = sqrt(A[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

template<class T>
__global__ void elementwiseOperationInPlace_kernel_div(T *A, const T* B, uint64_t arrayLength)
{
	uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (tid < arrayLength)
	{
		A[tid] = A[tid] / B[tid];
		tid += blockDim.x * gridDim.x;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel and input array storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_KernelAx[MAX_KERNEL_LENGTH];
__constant__ float c_KernelLat[MAX_KERNEL_LENGTH];
__constant__ float c_KernelEle[MAX_KERNEL_LENGTH];

extern "C" void setConvolutionKernelAx(float *h_Kernel, size_t kernel_length){
	if (kernel_length > MAX_KERNEL_LENGTH)
	{
		printf("ERROR: setConvolutionKernel : kernel length %d is too long. Change variable MAX_KERNEL_LENGTH and recompile code\n", kernel_length);
		exit(10);
	}
	cudaMemcpyToSymbol(c_KernelAx, h_Kernel, kernel_length  * sizeof(float));
}
extern "C" void setConvolutionKernelLat(float *h_Kernel, size_t kernel_length){
	if (kernel_length > MAX_KERNEL_LENGTH)
	{
		printf("ERROR: setConvolutionKernel : kernel length %d is too long. Change variable MAX_KERNEL_LENGTH and recompile code\n", kernel_length);
		exit(10);
	}
	cudaMemcpyToSymbol(c_KernelLat, h_Kernel, kernel_length * sizeof(float));
}
extern "C" void setConvolutionKernelEle(float *h_Kernel, size_t kernel_length){
	if (kernel_length > MAX_KERNEL_LENGTH)
	{
		printf("ERROR: setConvolutionKernel : kernel length %d is too long. Change variable MAX_KERNEL_LENGTH and recompile code\n", kernel_length);
		exit(10);
	}
	cudaMemcpyToSymbol(c_KernelEle, h_Kernel, kernel_length * sizeof(float));
}

texture<float, 3, cudaReadModeElementType> texSrc;

//extern "C" void setInputArray(cudaArray *a_Src){
//}
//
//extern "C" void detachInputArray(void){
//}




////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////

//using texture memory and step_size != 1
__global__ void convolutionRowsKernel(
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius,
	float step_size
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int idx = IMAD(iy, imageW, ix);
	const   int stride = imageW * imageH;
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;
		const float z = (float)iz + 0.5f;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
			sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelLat[kernel_radius - k];

		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}

//using global memory instead of texture memory
__global__ void convolutionRowsKernel(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelLat[kernel_radius - k];
			if (ix + k < 0)
				sum += d_Src[imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
			else if (ix + k >= imageW)
				sum += d_Src[imageW - 1 + imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
			else
				sum += d_Src[ix + k + imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}

//using global memory instead of texture memory
__global__ void convolutionRowsKernel_square(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelLat[kernel_radius - k];
			if (ix + k < 0)
				sum += d_Src[imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
			else if (ix + k >= imageW)
				sum += d_Src[imageW - 1 + imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
			else
				sum += d_Src[ix + k + imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] += (sum*sum);
	}
}
//using global memory instead of texture memory
__global__ void convolutionRowsKernel_add(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelLat[kernel_radius - k];
			if (ix + k < 0)
				sum += d_Src[imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
			else if (ix + k >= imageW)
				sum += d_Src[imageW - 1 + imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
			else
				sum += d_Src[ix + k + imageW * (iy + imageH * iz)] * c_KernelLat[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] += sum;
	}
}

//using texture memory and step_size = 1
__global__ void convolutionRowsKernel(
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int idx = IMAD(iy, imageW, ix);
	const   int stride = imageW * imageH;
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;
		const float z = (float)iz + 0.5f;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
			sum += tex3D(texSrc, x + (float)k, y, z) * c_KernelLat[kernel_radius - k];

		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}


extern "C" void convolutionRowsGPUtexture(
	float *d_Dst,
	cudaArray *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius,
	float step_size
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));


	checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
	texSrc.normalized = false;
	texSrc.addressMode[0] = cudaAddressModeClamp;
	texSrc.addressMode[1] = cudaAddressModeClamp;
	texSrc.addressMode[2] = cudaAddressModeClamp;

	if (step_size == 1.0f)
	{
		convolutionRowsKernel << <blocks, threads >> >(
			d_Dst,
			imageW,
			imageH,
			imageD,
			kernel_radius
			);
	}
	else{
		convolutionRowsKernel << <blocks, threads >> >(
			d_Dst,
			imageW,
			imageH,
			imageD,
			kernel_radius,
			step_size
			);
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionRowsKernel() execution failed\n");
	}

	checkCudaErrors(cudaUnbindTexture(texSrc));

}

extern "C" void convolutionRowsGPU(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));



	convolutionRowsKernel << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionRowsKernel() execution failed\n");
	}
}

extern "C" void convolutionRowsGPU_square(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));



	convolutionRowsKernel_square << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionRowsKernel() execution failed\n");
	}
}

extern "C" void convolutionRowsGPU_add(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));



	convolutionRowsKernel_add << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionRowsKernel() execution failed\n");
	}
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius,
	float step_size
	){
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int idx = IMAD(iy, imageW, ix);
	const   int stride = imageW * imageH;
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;
		const float z = (float)iz + 0.5f;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
			sum += tex3D(texSrc, x, y + step_size * (float)k, z) * c_KernelAx[kernel_radius - k];

		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}


__global__ void convolutionColumnsKernel(
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	){
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int idx = IMAD(iy, imageW, ix);
	const   int stride = imageW * imageH;
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;
		const float z = (float)iz + 0.5f;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
			sum += tex3D(texSrc, x, y + (float)k, z) * c_KernelAx[kernel_radius - k];

		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}


//using global memory instead of texture memory
__global__ void convolutionColumnsKernel(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelAx[kernel_radius - k];
			if (iy + k < 0)
				sum += d_Src[ix + imageW * (imageH * iz)] * c_KernelAx[kernel_radius - k];
			else if (iy + k >= imageH)
				sum += d_Src[ix + imageW * (imageH - 1 + imageH * iz)] * c_KernelAx[kernel_radius - k];
			else
				sum += d_Src[ix + imageW * (iy + k + imageH * iz)] * c_KernelAx[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}

//using global memory instead of texture memory
__global__ void convolutionColumnsKernel_square(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelAx[kernel_radius - k];
			if (iy + k < 0)
				sum += d_Src[ix + imageW * (imageH * iz)] * c_KernelAx[kernel_radius - k];
			else if (iy + k >= imageH)
				sum += d_Src[ix + imageW * (imageH - 1 + imageH * iz)] * c_KernelAx[kernel_radius - k];
			else
				sum += d_Src[ix + imageW * (iy + k + imageH * iz)] * c_KernelAx[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] += (sum*sum);
	}
}

//using global memory instead of texture memory
__global__ void convolutionColumnsKernel_add(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelAx[kernel_radius - k];
			if (iy + k < 0)
				sum += d_Src[ix + imageW * (imageH * iz)] * c_KernelAx[kernel_radius - k];
			else if (iy + k >= imageH)
				sum += d_Src[ix + imageW * (imageH - 1 + imageH * iz)] * c_KernelAx[kernel_radius - k];
			else
				sum += d_Src[ix + imageW * (iy + k + imageH * iz)] * c_KernelAx[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] += sum;
	}
}

extern "C" void convolutionColumnsGPUtexture(
	float *d_Dst,
	cudaArray *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius,
	float step_size
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

	checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));

	if (step_size == 1.0f)
	{
		convolutionColumnsKernel << <blocks, threads >> >(
			d_Dst,
			imageW,
			imageH,
			imageD,
			kernel_radius
			);
	}
	else{
		convolutionColumnsKernel << <blocks, threads >> >(
			d_Dst,
			imageW,
			imageH,
			imageD,
			kernel_radius,
			step_size
			);
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionColumnsKernel() execution failed\n");
	}

	checkCudaErrors(cudaUnbindTexture(texSrc));
}

extern "C" void convolutionColumnsGPU(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));


	convolutionColumnsKernel << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionColumnsKernel() execution failed\n");
	}


}

extern "C" void convolutionColumnsGPU_square(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));


	convolutionColumnsKernel_square << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionColumnsKernel() execution failed\n");
	}


}

extern "C" void convolutionColumnsGPU_add(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));


	convolutionColumnsKernel_add << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionColumnsKernel() execution failed\n");
	}


}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionDepthKernel(
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	){
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int idx = IMAD(iy, imageW, ix);
	const   int stride = imageW * imageH;
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;
		const float z = (float)iz + 0.5f;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
			sum += tex3D(texSrc, x, y, z + (float)k) * c_KernelEle[kernel_radius - k];

		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}

__global__ void convolutionDepthKernel(
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius,
	float step_size
	){
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int idx = IMAD(iy, imageW, ix);
	const   int stride = imageW * imageH;
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;
		const float z = (float)iz + 0.5f;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
			sum += tex3D(texSrc, x, y, z + step_size * (float)k) * c_KernelEle[kernel_radius - k];

		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}


//using global memory instead of texture memory
__global__ void convolutionDepthKernel(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelEle[kernel_radius - k];
			if (iz + k < 0)
				sum += d_Src[ix + imageW * iy] * c_KernelEle[kernel_radius - k];
			else if (iz + k >= imageD)
				sum += d_Src[ix + imageW * (iy + imageH * (imageD - 1))] * c_KernelEle[kernel_radius - k];
			else
				sum += d_Src[ix + imageW * (iy + imageH * (iz + k))] * c_KernelEle[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] = sum;
	}
}

//using global memory instead of texture memory
__global__ void convolutionDepthKernel_square(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelEle[kernel_radius - k];
			if (iz + k < 0)
				sum += d_Src[ix + imageW * iy] * c_KernelEle[kernel_radius - k];
			else if (iz + k >= imageD)
				sum += d_Src[ix + imageW * (iy + imageH * (imageD - 1))] * c_KernelEle[kernel_radius - k];
			else
				sum += d_Src[ix + imageW * (iy + imageH * (iz + k))] * c_KernelEle[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] += (sum * sum);
	}
}


//using global memory instead of texture memory
__global__ void convolutionDepthKernel_add(
	const float *d_Src,
	float *d_Dst,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const   int stride = imageW * imageH;


	if (ix >= imageW || iy >= imageH)
		return;


	for (int iz = 0; iz < imageD; iz++)
	{
		float sum = 0;

		for (int k = -kernel_radius; k <= kernel_radius; k++)
		{
			//sum += tex3D(texSrc, x + step_size * (float)k, y, z) * c_KernelEle[kernel_radius - k];
			if (iz + k < 0)
				sum += d_Src[ix + imageW * iy] * c_KernelEle[kernel_radius - k];
			else if (iz + k >= imageD)
				sum += d_Src[ix + imageW * (iy + imageH * (imageD - 1))] * c_KernelEle[kernel_radius - k];
			else
				sum += d_Src[ix + imageW * (iy + imageH * (iz + k))] * c_KernelEle[kernel_radius - k];
		}
		d_Dst[IMAD(iy, imageW, ix) + stride * iz] += sum;
	}
}
extern "C" void convolutionDepthGPUtexture(
	float *d_Dst,
	cudaArray *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius,
	float step_size
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

	checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));

	if (step_size == 1.0f)
	{
		convolutionDepthKernel << <blocks, threads >> >(
			d_Dst,
			imageW,
			imageH,
			imageD,
			kernel_radius
			);
	}
	else{
		convolutionDepthKernel << <blocks, threads >> >(
			d_Dst,
			imageW,
			imageH,
			imageD,
			kernel_radius,
			step_size
			);
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionDepthKernel() execution failed\n");
	}

	checkCudaErrors(cudaUnbindTexture(texSrc));
}


extern "C" void convolutionDepthGPU(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));


	convolutionDepthKernel << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionDepthKernel() execution failed\n");
	}
}

extern "C" void convolutionDepthGPU_square(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));


	convolutionDepthKernel_square << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionDepthKernel() execution failed\n");
	}
}

extern "C" void convolutionDepthGPU_add(
	float *d_Dst,
	const float *a_Src,
	int imageW,
	int imageH,
	int imageD,
	int kernel_radius
	)
{
	dim3 threads(16, 16, 1);
	dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));


	convolutionDepthKernel_add << <blocks, threads >> >(
		a_Src,
		d_Dst,
		imageW,
		imageH,
		imageD,
		kernel_radius
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		printf("convolutionDepthKernel() execution failed\n");
	}
}

////////////////////////////////////////////////////////////////////////////////
// Calculate Hessian using Gaussian derivatives and separable convolution
////////////////////////////////////////////////////////////////////////////////
//Hessian_CUDA should have been allocated as 6 * sizeof(float) * imSize
//Hessian_CUDA[ii*imSize, (ii+1) * imSize) constins Hessian for ii-th second-order derivatives
//stepSize[ii] = step for along each dimension. If data has anisotropy, we account for it in the derivative
extern "C" void HessianWithGaussianDerivativesGPU_texture(
	const float *img_HOST,
	float *Hessian_CUDA,
	int64_t imgDims[3],
	float sigma,
	const float step_size[3],
	int kernel_radius
	)
{
	int64_t numPixels = imgDims[0] * imgDims[1] * imgDims[2];

	//prepare texture memory
	cudaArray *a_Src;
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
	cudaExtent volExtent = make_cudaExtent(imgDims[0], imgDims[1], imgDims[2]);
	checkCudaErrors(cudaMalloc3DArray(&a_Src, &floatTex, volExtent));

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)img_HOST, imgDims[0] * sizeof(float), imgDims[0], imgDims[1]);
	copyParams.dstArray = a_Src;
	copyParams.extent = volExtent;
	copyParams.kind = cudaMemcpyHostToDevice;


	texSrc.normalized = false; //coordinates are not between [0,1]^dimsImage but between image size boundaries
	texSrc.filterMode = cudaFilterModeLinear;//use cudaFilterModePoint nearest neighbor interpolation. Use cudaFilterModeLinear for linear interpolation
	texSrc.addressMode[0] = cudaAddressModeClamp;//How out of bounds requests are handled. For non-normalized mode only clamp is supported. In clamp addressing mode x is replaced by 0 if x<0 and N-1 if x>=N;
	texSrc.addressMode[1] = cudaAddressModeClamp;
	texSrc.addressMode[2] = cudaAddressModeClamp;

	//calculate kernels for each order derivative
	const float sigma2 = sigma * sigma;
	const int kernel_length = 2 * kernel_radius + 1;
	float aux;
	float *kernelGaussian_d0 = (float*)malloc(sizeof(float)* kernel_length);
	float *kernelGaussian_d1 = (float*)malloc(sizeof(float)* kernel_length);
	float *kernelGaussian_d2 = (float*)malloc(sizeof(float)* kernel_length);

	for (int ii = -kernel_radius; ii <= kernel_radius; ii++)
	{
		aux = 0.3989422804014327f * exp(-0.5f * powf(float(ii), 2.0f) / sigma2) / sigma;//1/sqrt( 2.0 * 3.14159265358979311600f)= 0.3989422804014327; 
		kernelGaussian_d0[ii + kernel_radius] = aux;
		kernelGaussian_d1[ii + kernel_radius] = -aux * float(ii) / sigma2;
		kernelGaussian_d2[ii + kernel_radius] = aux * (float(ii*ii) / sigma2 - 1.0f) / sigma2;
	}


	//calculate Hessian	
	for (int ii = 0; ii < 6; ii++)
	{
		float *d_Output = &(Hessian_CUDA[numPixels * (int64_t)ii]);

		//reset image in the GPU
		copyParams.srcPtr = make_cudaPitchedPtr((void*)(img_HOST), volExtent.width*sizeof(float), volExtent.width, volExtent.height);
		copyParams.dstArray = a_Src;
		//copyParams.extent   = volExtent;
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));


		//upload kernels
		switch (ii)//the kernels that do not change from one iteration to the next are commented out
		{
		case 0:
			setConvolutionKernelLat(kernelGaussian_d2, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d0, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d0, kernel_length);
			break;

		case 1:
			setConvolutionKernelLat(kernelGaussian_d1, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d1, kernel_length);
			//setConvolutionKernelEle(kernelGaussian_d0, kernel_length);
			break;

		case 2:
			//setConvolutionKernelLat(kernelGaussian_d1, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d0, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d1, kernel_length);
			break;

		case 3:
			setConvolutionKernelLat(kernelGaussian_d0, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d2, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d0, kernel_length);
			break;

		case 4:
			//setConvolutionKernelLat(kernelGaussian_d0, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d1, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d1, kernel_length);
			break;

		case 5:
			//setConvolutionKernelLat(kernelGaussian_d0, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d0, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d2, kernel_length);
			break;

		default:
			printf("ERROR: code not ready for this value\n");

		}

		//calculate convolution
		convolutionRowsGPUtexture(d_Output, a_Src, imgDims[0], imgDims[1], imgDims[2], kernel_radius, step_size[0]);
		copyParams.srcPtr = make_cudaPitchedPtr((void*)(d_Output), volExtent.width*sizeof(float), volExtent.width, volExtent.height);
		copyParams.dstArray = a_Src;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));

		convolutionColumnsGPUtexture(d_Output, a_Src, imgDims[0], imgDims[1], imgDims[2], kernel_radius, step_size[1]);
		copyParams.srcPtr = make_cudaPitchedPtr((void*)(d_Output), volExtent.width*sizeof(float), volExtent.width, volExtent.height);
		copyParams.dstArray = a_Src;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));

		convolutionDepthGPUtexture(d_Output, a_Src, imgDims[0], imgDims[1], imgDims[2], kernel_radius, step_size[2]);
		//no need to copy output to CUDA array
	}

	//release memory
	free(kernelGaussian_d0);
	free(kernelGaussian_d1);
	free(kernelGaussian_d2);

	checkCudaErrors(cudaFreeArray(a_Src));
}

////////////////////////////////////////////////////////////////////////////////
// Calculate image blurring trhough a Gaussian separable kernel
////////////////////////////////////////////////////////////////////////////////
//Hessian_CUDA should have been allocated as 6 * sizeof(float) * imSize
//Hessian_CUDA[ii*imSize, (ii+1) * imSize) constins Hessian for ii-th second-order derivatives
//stepSize[ii] = step for along each dimension. If data has anisotropy, we account for it in the derivative

extern "C" void imgaussianAnisotropy(
	float *src_CUDA,
	float *dst_CUDA,
	int64_t imgDims[3],
	float sigma[3],
	int kernel_radius[3]
	)
{

	//calculate convolution for each directions
	for (int ii = 0; ii < 3; ii++)
	{

		//calculate kernels for each order derivative
		float sigma2 = sigma[ii] * sigma[ii];
		int kernel_length = 2 * kernel_radius[ii] + 1;
		float aux;
		float *kernelGaussian_d0 = (float*)malloc(sizeof(float)* kernel_length);

		for (int kk = -kernel_radius[ii]; kk <= kernel_radius[ii]; kk++)
		{
			aux = 0.3989422804014327f * exp(-0.5f * powf(float(kk), 2.0f) / sigma2) / sigma[ii];//1/sqrt( 2.0 * 3.14159265358979311600f)= 0.3989422804014327; 
			kernelGaussian_d0[kk + kernel_radius[ii]] = aux;
		}

		float *d_Output, *d_Input;
		if (ii % 2 == 0)
		{
			d_Output = dst_CUDA;
			d_Input = src_CUDA;
		}
		else{
			d_Output = src_CUDA;
			d_Input = dst_CUDA;
		}

		//upload kernels
		switch (ii)//the kernels that do not change from one iteration to the next are commented out
		{
		case 0:
			setConvolutionKernelLat(kernelGaussian_d0, kernel_length);
			convolutionRowsGPU(d_Output, d_Input, imgDims[0], imgDims[1], imgDims[2], kernel_radius[ii]);
			break;

		case 1:
			setConvolutionKernelAx(kernelGaussian_d0, kernel_length);
			convolutionColumnsGPU(d_Output, d_Input, imgDims[0], imgDims[1], imgDims[2], kernel_radius[ii]);
			break;

		case 2:
			setConvolutionKernelEle(kernelGaussian_d0, kernel_length);
			convolutionDepthGPU(d_Output, d_Input, imgDims[0], imgDims[1], imgDims[2], kernel_radius[ii]);
			break;

		default:
			printf("ERROR: code not ready for this value\n");

		}

		//release memory
		free(kernelGaussian_d0);
	}

}

////////////////////////////////////////////////////////////////////////////////
// Calculate Hessian using Gaussian derivatives and separable convolution
//Special case when only Z has anisotropy. We can avoid many copies
//TOD0: write a function with no anisotropy that does not require texture memory
////////////////////////////////////////////////////////////////////////////////
//Hessian_CUDA should have been allocated as 6 * sizeof(float) * imSize
//Hessian_CUDA[ii*imSize, (ii+1) * imSize) constins Hessian for ii-th second-order derivatives
//stepSize[ii] = step for along each dimension. If data has anisotropy, we account for it in the derivative
extern "C" void HessianWithGaussianDerivativesGPU_AnisotropyZ(
	const float *img_HOST,
	float *Hessian_CUDA,
	int64_t imgDims[3],
	float sigma,
	const float step_size_z,
	int kernel_radius
	)
{
	int64_t numPixels = imgDims[0] * imgDims[1] * imgDims[2];

	//prepare texture memory
	cudaArray *a_Src;
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
	cudaExtent volExtent = make_cudaExtent(imgDims[0], imgDims[1], imgDims[2]);
	checkCudaErrors(cudaMalloc3DArray(&a_Src, &floatTex, volExtent));

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)img_HOST, imgDims[0] * sizeof(float), imgDims[0], imgDims[1]);
	copyParams.dstArray = a_Src;
	copyParams.extent = volExtent;
	copyParams.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&copyParams));//we copy only once



	texSrc.normalized = false; //coordinates are not between [0,1]^dimsImage but between image size boundaries
	texSrc.filterMode = cudaFilterModeLinear;//use cudaFilterModePoint nearest neighbor interpolation. Use cudaFilterModeLinear for linear interpolation
	texSrc.addressMode[0] = cudaAddressModeClamp;//How out of bounds requests are handled. For non-normalized mode only clamp is supported. In clamp addressing mode x is replaced by 0 if x<0 and N-1 if x>=N;
	texSrc.addressMode[1] = cudaAddressModeClamp;
	texSrc.addressMode[2] = cudaAddressModeClamp;

	//we need one more auxiliary copy of the image for the out-of-place convolution
	float *d_output_swap;
	checkCudaErrors(cudaMalloc((void**)&(d_output_swap), numPixels * sizeof(float)));


	//calculate kernels for each order derivative
	const float sigma2 = sigma * sigma;
	const int kernel_length = 2 * kernel_radius + 1;
	float aux;
	float *kernelGaussian_d0 = (float*)malloc(sizeof(float)* kernel_length);
	float *kernelGaussian_d1 = (float*)malloc(sizeof(float)* kernel_length);
	float *kernelGaussian_d2 = (float*)malloc(sizeof(float)* kernel_length);

	for (int ii = -kernel_radius; ii <= kernel_radius; ii++)
	{
		aux = 0.3989422804014327f * exp(-0.5f * powf(float(ii), 2.0f) / sigma2) / sigma;//1/sqrt( 2.0 * 3.14159265358979311600f)= 0.3989422804014327; 
		kernelGaussian_d0[ii + kernel_radius] = aux;
		kernelGaussian_d1[ii + kernel_radius] = -aux * float(ii) / sigma2;
		kernelGaussian_d2[ii + kernel_radius] = aux * (float(ii*ii) / sigma2 - 1.0f) / sigma2;
	}


	//calculate Hessian	
	for (int ii = 0; ii < 6; ii++)
	{
		float *d_Output = &(Hessian_CUDA[numPixels * (int64_t)ii]);

		//upload kernels
		switch (ii)//the kernels that do not change from one iteration to the next are commented out
		{
		case 0:
			setConvolutionKernelLat(kernelGaussian_d2, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d0, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d0, kernel_length);
			break;

		case 1:
			setConvolutionKernelLat(kernelGaussian_d1, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d1, kernel_length);
			//setConvolutionKernelEle(kernelGaussian_d0, kernel_length);
			break;

		case 2:
			//setConvolutionKernelLat(kernelGaussian_d1, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d0, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d1, kernel_length);
			break;

		case 3:
			setConvolutionKernelLat(kernelGaussian_d0, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d2, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d0, kernel_length);
			break;

		case 4:
			//setConvolutionKernelLat(kernelGaussian_d0, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d1, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d1, kernel_length);
			break;

		case 5:
			//setConvolutionKernelLat(kernelGaussian_d0, kernel_length);
			setConvolutionKernelAx(kernelGaussian_d0, kernel_length);
			setConvolutionKernelEle(kernelGaussian_d2, kernel_length);
			break;

		default:
			printf("ERROR: code not ready for this value\n");

		}

		//calculate convolution

		//first in Z direction because we require interpolation and the texture memory
		convolutionDepthGPUtexture(d_Output, a_Src, imgDims[0], imgDims[1], imgDims[2], kernel_radius, step_size_z);

		//now x and y direction are done in the global memory (they do not require interpolation)
		convolutionRowsGPU(d_output_swap, d_Output, imgDims[0], imgDims[1], imgDims[2], kernel_radius);


		convolutionColumnsGPU(d_Output, d_output_swap, imgDims[0], imgDims[1], imgDims[2], kernel_radius);

	}

	//release memory
	free(kernelGaussian_d0);
	free(kernelGaussian_d1);
	free(kernelGaussian_d2);

	checkCudaErrors(cudaFreeArray(a_Src));
	checkCudaErrors(cudaFree(d_output_swap));
}



//======================================================================
extern "C" void TV_gradient_norm(const float *img_CUDA, float *TV_grad_norm_CUDA, int64_t imgDims[3], float sigma, int kernel_radius)
{
	//calculate kernels for each first derivative
	const float sigma2 = sigma * sigma;
	const int kernel_length = 2 * kernel_radius + 1;
	float aux;
	float *kernelGaussian_d1 = (float*)malloc(sizeof(float)* kernel_length); //first derivative with Gaussian smoothing	

	for (int ii = -kernel_radius; ii <= kernel_radius; ii++)
	{
		aux = 0.3989422804014327f * exp(-0.5f * powf(float(ii), 2.0f) / sigma2) / sigma;//1/sqrt( 2.0 * 3.14159265358979311600f)= 0.3989422804014327; 		
		kernelGaussian_d1[ii + kernel_radius] = -aux * float(ii) / sigma2;
	}


	//set kernels
	setConvolutionKernelLat(kernelGaussian_d1, kernel_length);
	setConvolutionKernelAx(kernelGaussian_d1, kernel_length);
	setConvolutionKernelEle(kernelGaussian_d1, kernel_length);

	//set norm to zero
	uint64_t arrayLength = imgDims[0] * imgDims[1] * imgDims[2];
	HANDLE_ERROR(cudaMemset(TV_grad_norm_CUDA, 0, arrayLength * sizeof(float)));

	//compute the three derivatives
	convolutionRowsGPU_square(TV_grad_norm_CUDA, img_CUDA, imgDims[0], imgDims[1], imgDims[2], kernel_radius);
	convolutionColumnsGPU_square(TV_grad_norm_CUDA, img_CUDA, imgDims[0], imgDims[1], imgDims[2], kernel_radius);
	convolutionDepthGPU_square(TV_grad_norm_CUDA, img_CUDA, imgDims[0], imgDims[1], imgDims[2], kernel_radius);

	//take square root
#if __CUDA_ARCH__ < 300	
	int MAX_BLOCKS_CUDA = 65535;
#else	
	int MAX_BLOCKS_CUDA = 2147483647;
#endif

	int numThreads = std::min((uint64_t)256, arrayLength);//profiling it is better to not use all threads for better occupancy
	int numBlocks = std::min((uint64_t)MAX_BLOCKS_CUDA, (uint64_t)(arrayLength + (uint64_t)(numThreads - 1)) / ((uint64_t)numThreads));

	elementwiseOperationInPlace_kernel_sqRoot << <numBlocks, numThreads >> >(TV_grad_norm_CUDA, arrayLength); HANDLE_ERROR_KERNEL;

#if _DEBUG
	std::string filenameDebug("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/debug_TV_grad_norm.raw");
	float* TV_grad_norm_CPU = new float[arrayLength];
	HANDLE_ERROR(cudaMemcpy(TV_grad_norm_CPU, TV_grad_norm_CUDA, arrayLength * sizeof(float), cudaMemcpyDeviceToHost));
	FILE* fid = fopen(filenameDebug.c_str(), "wb");
	if (fid == NULL)
	{
		printf("Error opening file %s to save raw image data\n", filenameDebug.c_str());
		return;
	}
	else{
		printf("====DEBUGGING:Write debug TV_grad_norm_CPU to file %s================\n", filenameDebug.c_str());
	}
	fwrite(TV_grad_norm_CPU, sizeof(float), arrayLength, fid);
	fclose(fid);

	delete[] TV_grad_norm_CPU;

	//original image
	filenameDebug = std::string("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/debug_TV_img_CUDA.raw");
	TV_grad_norm_CPU = new float[arrayLength];
	HANDLE_ERROR(cudaMemcpy(TV_grad_norm_CPU, img_CUDA, arrayLength * sizeof(float), cudaMemcpyDeviceToHost));
	fid = fopen(filenameDebug.c_str(), "wb");
	if (fid == NULL)
	{
		printf("Error opening file %s to save raw image data\n", filenameDebug.c_str());
		return;
	}
	else{
		printf("====DEBUGGING:Write debug TV_grad_norm_CPU to file %s================\n", filenameDebug.c_str());
	}
	fwrite(TV_grad_norm_CPU, sizeof(float), arrayLength, fid);
	fclose(fid);

	delete[] TV_grad_norm_CPU;
#endif

	//release memory	
	free(kernelGaussian_d1);
}

//======================================================================
extern "C" void TV_divergence(
	const float *img_CUDA,
	const float *TV_grad_norm_CUDA,
	float* temp_CUDA,
	float *TV_out,
	int64_t imgDims[3],
	float sigma,
	int kernel_radius
	)
{

	//calculate kernel for first order derivative
	const float sigma2 = sigma * sigma;
	const int kernel_length = 2 * kernel_radius + 1;
	float aux;
	float *kernelGaussian_d1 = (float*)malloc(sizeof(float)* kernel_length); //first derivative with Gaussian smoothing	

	for (int ii = -kernel_radius; ii <= kernel_radius; ii++)
	{
		aux = 0.3989422804014327f * exp(-0.5f * powf(float(ii), 2.0f) / sigma2) / sigma;//1/sqrt( 2.0 * 3.14159265358979311600f)= 0.3989422804014327; 		
		kernelGaussian_d1[ii + kernel_radius] = -aux * float(ii) / sigma2;
	}

	//kernel for finite difference
	const int finite_diff_radius = 1;
	const int finite_diff_length = 2 * finite_diff_radius + 1;
	float *kernel_finite_diff = (float*)malloc(sizeof(float)* finite_diff_length);
	kernel_finite_diff[0] = -1;
	kernel_finite_diff[1] = 0;
	kernel_finite_diff[2] = 1;


	//constants for element wise operation
	uint64_t arrayLength = imgDims[0] * imgDims[1] * imgDims[2];
#if __CUDA_ARCH__ < 300	
	int MAX_BLOCKS_CUDA = 65535;
#else	
	int MAX_BLOCKS_CUDA = 2147483647;
#endif

	int numThreads = std::min((uint64_t)256, arrayLength);//profiling it is better to not use all threads for better occupancy
	int numBlocks = std::min((uint64_t)MAX_BLOCKS_CUDA, (uint64_t)(arrayLength + (uint64_t)(numThreads - 1)) / ((uint64_t)numThreads));

	//set divergence to zero	
	HANDLE_ERROR(cudaMemset(TV_out, 0, arrayLength * sizeof(float)));

	//calculate divergence across dimension x
	setConvolutionKernelLat(kernelGaussian_d1, kernel_length);
	convolutionRowsGPU(temp_CUDA, img_CUDA, imgDims[0], imgDims[1], imgDims[2], kernel_radius);//partial derivative
	elementwiseOperationInPlace_kernel_div << <numBlocks, numThreads >> >(temp_CUDA, TV_grad_norm_CUDA, arrayLength); HANDLE_ERROR_KERNEL;//divide by norm of teh gradient
	setConvolutionKernelLat(kernel_finite_diff, finite_diff_length);//finite differences for divergence
	convolutionRowsGPU_add(TV_out, temp_CUDA, imgDims[0], imgDims[1], imgDims[2], finite_diff_radius);


	//calculate divergence across dimension y
	setConvolutionKernelAx(kernelGaussian_d1, kernel_length);
	convolutionColumnsGPU(temp_CUDA, img_CUDA, imgDims[0], imgDims[1], imgDims[2], kernel_radius);//partial derivative
	elementwiseOperationInPlace_kernel_div << <numBlocks, numThreads >> >(temp_CUDA, TV_grad_norm_CUDA, arrayLength); HANDLE_ERROR_KERNEL;//divide by norm of teh gradient
	setConvolutionKernelAx(kernel_finite_diff, finite_diff_length);//finite differences for divergence
	convolutionColumnsGPU_add(TV_out, temp_CUDA, imgDims[0], imgDims[1], imgDims[2], finite_diff_radius);

	//calculate divergence across dimension z
	setConvolutionKernelEle(kernelGaussian_d1, kernel_length);
	convolutionDepthGPU(temp_CUDA, img_CUDA, imgDims[0], imgDims[1], imgDims[2], kernel_radius);//partial derivative
	elementwiseOperationInPlace_kernel_div << <numBlocks, numThreads >> >(temp_CUDA, TV_grad_norm_CUDA, arrayLength); HANDLE_ERROR_KERNEL;//divide by norm of teh gradient
	setConvolutionKernelEle(kernel_finite_diff, finite_diff_length);//finite differences for divergence
	convolutionDepthGPU_add(TV_out, temp_CUDA, imgDims[0], imgDims[1], imgDims[2], finite_diff_radius);


	//release memory	
	free(kernelGaussian_d1);
	free(kernel_finite_diff);
}