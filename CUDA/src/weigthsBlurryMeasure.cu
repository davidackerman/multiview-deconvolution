/*
* weightsBlurryMeasure.h
*
*  Created on: Jul 20, 2015
*      Author: Fernando Amat
*/

#include <assert.h>
#include <math.h>
#include "book.h"
#include "cuda.h"
#include "commonCUDA.h"
#include "weigthsBlurryMeasure.h"
#include "dct8x8_kernel2.cuh"
#include "convolutionTexture_common.h"



/**
**************************************************************************
*  Computes shannon entropy of DCT coefficients (measurement of contrast)
*	(based on CUDAkernelQuantizationFloat kernel from Nvidia samples)
*
* \param SrcDst         [IN/OUT] - DCT coefficients plane
* \param Stride         [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernelDCTShannonEntropyFloat(float *SrcDst, int Stride, float cutoff)
{
	__shared__ float DCTcoeff[BLOCK_SIZE2];
	__shared__ float DCTcoeffAbs[BLOCK_SIZE2];
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index (current coefficient)
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int cacheIndex = tx + blockDim.x * ty;

	//copy current coefficient to the local variable and calculate entropy for each entry		
	if (sqrtf(float((tx+1)*(tx+1) + (ty+1)*(ty+1))) > cutoff)//+1 to match Matlab code
		DCTcoeff[cacheIndex] = 0;//to avoid having to check for 0
	else	
		DCTcoeff[cacheIndex] = fabs(SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)]);
	//copy absolute value
	DCTcoeffAbs[cacheIndex] = powf(DCTcoeff[cacheIndex], 2);
	__syncthreads();

	
	//calculate norm	
	int i = blockDim.x * blockDim.y / 2;	
	while (i != 0)
	{
		if (cacheIndex < i) 
			DCTcoeffAbs[cacheIndex] += DCTcoeffAbs[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	//float norm = DCTcoeffAbs[0];
	//__syncthreads();

	//calculate entropy term for each elements
	DCTcoeff[cacheIndex] /= sqrtf(DCTcoeffAbs[0]);
	DCTcoeff[cacheIndex] = DCTcoeff[cacheIndex] * log2(DCTcoeff[cacheIndex] + 1e-7);//add epsilon to avoid 0 * inf = NaN
	__syncthreads();

	//calculate entropy
	i = blockDim.x * blockDim.y / 2;	
	while (i != 0)
	{
		if (cacheIndex < i)
			DCTcoeff[cacheIndex] += DCTcoeff[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	//copy quantized coefficient back to the DCT-plane
	SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)] = -DCTcoeff[0];
}


//================================================================

void calculateWeightsDeconvolution(float* weights_CUDA, float* img_CUDA, int64_t *dims, int ndims, float anisotropyZ, bool normalize, float weightsPower, float weightsThreshold)
{
	assert(ndims <= 3);
	assert(ndims > 1);

	int64_t imSize = 1;
	for (int ii = 0; ii < ndims; ii++)
		imSize *= dims[ii];
	
	float *temp_CUDA;//we need temporary copy for out-of-place separable convolution
	HANDLE_ERROR(cudaMalloc((void**)&(temp_CUDA), imSize * sizeof(float)));
	
	//calculate DCT for each plane
	//setup execution parameters
	dim3 GridFullWarps(dims[0] / KER2_BLOCK_WIDTH, dims[1] / KER2_BLOCK_HEIGHT, 1);
	dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH / 8, KER2_BLOCK_HEIGHT / 8);

	//setup parameters
	int numPlanes = 1;
	if (ndims > 2)
		numPlanes = dims[2];	

	float *dst = temp_CUDA, *src = img_CUDA;
	const int DeviceStride = dims[0];
	const int64_t SliceOffset = dims[0] * dims[1];

	//perform block-wise DCT processing and benchmarking
	int64_t offset = 0;		
	for (int64_t i = 0; i < numPlanes; i++)
	{	
		CUDAkernel2DCT << <GridFullWarps, ThreadsFullWarps >> >(dst, src, DeviceStride); HANDLE_ERROR_KERNEL;		
		offset += SliceOffset;
		dst = &(temp_CUDA[offset]);
		src = &(img_CUDA[offset]);
	}
			
	
	//setup execution parameters for shannon entropy
	dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
	dim3 GridSmallBlocks(dims[0] / BLOCK_SIZE, dims[1] / BLOCK_SIZE);

	// execute shannon entropy kernel
	dst = temp_CUDA;
	offset = 0;
	for (int64_t i = 0; i < numPlanes; i++)
	{
		CUDAkernelDCTShannonEntropyFloat << < GridSmallBlocks, ThreadsSmallBlocks >> >(dst, DeviceStride, cutoffDCT); HANDLE_ERROR_KERNEL;		
		offset += SliceOffset;
		dst = &(temp_CUDA[offset]);
	}


	//printf("============DEBUGGING: not applying blur==============\n");
	//HANDLE_ERROR(cudaMemcpy(weights_CUDA, temp_CUDA, imSize * sizeof(float), cudaMemcpyDeviceToDevice));
	
	
	//apply seprable Gaussian kernel for smoothing
	float sigma[3] = { 0.5f * cellDiameterPixels, 0.5f * cellDiameterPixels, 0.5f * cellDiameterPixels / anisotropyZ };
	int kernel_radius[3];
	for (int ii = 0; ii < 3; ii++)
	{
		kernel_radius[ii] = ceil(5.0f * sigma[ii]);
	}
		
	imgaussianAnisotropy(temp_CUDA, weights_CUDA, dims, sigma, kernel_radius);
	
	
	//normalize weights
	if (normalize)
	{

		float minW = reductionOperation(weights_CUDA, imSize, op_reduction_type::min_elem);
		float maxW = reductionOperation(weights_CUDA, imSize, op_reduction_type::max_elem);
		elementwiseOperationInPlace(weights_CUDA, minW, imSize, op_elementwise_type::minus);
		elementwiseOperationInPlace(weights_CUDA, maxW - minW, imSize, op_elementwise_type::divide);

		if (weightsPower != 1.0f)
		{			
			elementwiseOperationInPlace(weights_CUDA, weightsPower, imSize, op_elementwise_type::power);
		}		
		if (weightsThreshold > 0.0f)
		{			
			elementwiseOperationInPlace(weights_CUDA, weightsThreshold, imSize, op_elementwise_type::threshold);
		}
	}
	//release memory
	HANDLE_ERROR(cudaFree(temp_CUDA));
	
}
