/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  multivieDeconvolution.cpp
*
*  Created on : June 5th, 2015
* Author : Fernando Amat
*
* \brief main interface to execute multiview deconvolution (it has ome abstract methods)
*/

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <fstream>
#include "multiviewDeconvolution.h"
#include "book.h"
#include "cuda.h"
#include "cufft.h"
#include "commonCUDA.h"


using namespace std;

//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
template<class imageType>
__global__ void __launch_bounds__(MAX_THREADS_CUDA) fftShiftKernel(imageType* kernelCUDA, imageType* kernelPaddedCUDA, int kernelDim_0, int kernelDim_1, int kernelDim_2, int imDim_0, int imDim_1, int imDim_2)
{
	int kernelSize = kernelDim_0 * kernelDim_1 * kernelDim_2;

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid<kernelSize)
	{
		//find coordinates
		int64_t x, y, z, aux;
		z = tid % kernelDim_2;
		aux = (tid - z) / kernelDim_2;
		y = aux % kernelDim_1;
		x = (aux - y) / kernelDim_1;

		//center coordinates
		x -= kernelDim_0 / 2;
		y -= kernelDim_1 / 2;
		z -= kernelDim_2 / 2;

		//circular shift if necessary
		if (x<0) x += imDim_0;
		if (y<0) y += imDim_1;
		if (z<0) z += imDim_2;

		//calculate position in padded kernel
		aux = z + imDim_2 * (y + imDim_1 * x);

		//copy value
		kernelPaddedCUDA[aux] = kernelCUDA[tid];//for the most part it should be a coalescent access in both places
	}
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
//Adapted from CUDA SDK examples
__device__ void mulAndScale(cufftComplex& a, const cufftComplex& b, const float& c)
{
	cufftComplex t = { c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y) };
	a = t;
};

//we multiply by conj(b) = {b.x, -b.y}
__device__ void mulAndScale_conj(cufftComplex& a, const cufftComplex& b, const float& c)
{
	cufftComplex t = { c * (a.x * b.x + a.y * b.y), c * (a.y * b.x - a.x * b.y) };
	a = t;
};

__device__ cufftComplex mulAndScale_outOfPlace(const cufftComplex& a, const cufftComplex& b, const float& c)
{
	return  { c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y) };	
};

__global__ void modulateAndNormalize_kernel(cufftComplex *d_Dst, const cufftComplex *d_Src, long long int dataSize, float c)
{
	std::int64_t i = (std::int64_t)blockDim.x * (std::int64_t)blockIdx.x + (std::int64_t)threadIdx.x;
	std::int64_t offset = (std::int64_t)blockDim.x * (std::int64_t)gridDim.x;
	while (i < dataSize)
	{
		//TODO: try speed difference without intermediate variables
		cufftComplex a = d_Src[i];
		cufftComplex b = d_Dst[i];
		mulAndScale(b, a, c);
		d_Dst[i] = b;

		i += offset;
	}
};

__global__ void modulateAndNormalize_conj_kernel(cufftComplex *d_Dst, const cufftComplex *d_Src, long long int dataSize, float c)
{
	std::int64_t i = (std::int64_t)blockDim.x * (std::int64_t)blockIdx.x + (std::int64_t)threadIdx.x;
	std::int64_t offset = (std::int64_t)blockDim.x * (std::int64_t)gridDim.x;
	while (i < dataSize)
	{
		//TODO: try speed difference without intermediate variables
		cufftComplex a = d_Src[i];
		cufftComplex b = d_Dst[i];
		mulAndScale_conj(b, a, c);
		d_Dst[i] = b;

		i += offset;
	}
};

__global__ void modulateAndNormalize_outOfPlace_kernel(cufftComplex *d_Dst, const cufftComplex *d_Src1, const cufftComplex *d_Src2, long long int dataSize, float c)
{
	std::int64_t i = (std::int64_t)blockDim.x * (std::int64_t)blockIdx.x + (std::int64_t)threadIdx.x;
	std::int64_t offset = (std::int64_t)blockDim.x * (std::int64_t)gridDim.x;
	while (i < dataSize)
	{
		d_Dst[i] = mulAndScale_outOfPlace(d_Src1[i], d_Src2[i], c);
		i += offset;
	}
};
//===========================================================================

template<class imgType>
multiviewDeconvolution<imgType>::multiviewDeconvolution()
{
	J.resize(1);//allocate for the output	
	fftPlanInv = -1;
	fftPlanFwd = -1;
}

//=======================================================
template<class imgType>
multiviewDeconvolution<imgType>::~multiviewDeconvolution()
{
	if (fftPlanInv >= 0)
	{
		(cufftDestroy(fftPlanInv)); HANDLE_ERROR_KERNEL;
	}
	if (fftPlanFwd >= 0)
	{
		(cufftDestroy(fftPlanFwd)); HANDLE_ERROR_KERNEL;
	}
}


//=======================================================
template<class imgType>
void multiviewDeconvolution<imgType>::setNumberOfViews(int numViews)
{
	weights.resize(numViews);
	psf.resize(numViews);
	img.resize(numViews);
}

//=======================================================
template<class imgType>
int multiviewDeconvolution<imgType>::readImage(const std::string& filename, int pos, const std::string& type)
{
	if (type.compare("weight") == 0)
		return weights.readImage(filename, pos);
	else if (type.compare("psf") == 0)
		return psf.readImage(filename, pos);
	else if (type.compare("img") == 0)
		return img.readImage(filename, pos);
	
	cout << "ERROR: multiviewDeconvolution<imgType>::readImage :option " << type << " not recognized" << endl;
	return 3;
}

//=======================================================
template<class imgType>
void multiviewDeconvolution<imgType>::padArrayWithZeros(const std::uint32_t *dimsAfterPad, int pos, const std::string& type)
{
	if (type.compare("weight") == 0)
		return weights.padArrayWithZeros(pos, dimsAfterPad);
	else if (type.compare("psf") == 0)
		return psf.padArrayWithZeros(pos, dimsAfterPad);
	else if (type.compare("img") == 0)
		return img.padArrayWithZeros(pos, dimsAfterPad);

	cout << "ERROR: multiviewDeconvolution<imgType>::readImage :option " << type << " not recognized" << endl;	
}


//=======================================================
template<class imgType>
int multiviewDeconvolution<imgType>::readROI(const std::string& filename, int pos, const std::string& type, const klb_ROI& ROI)
{
	if (type.compare("weight") == 0)
		return weights.readROI(filename, pos, ROI);
	else if (type.compare("psf") == 0)
		return psf.readROI(filename, pos, ROI);
	else if (type.compare("img") == 0)
		return img.readROI(filename, pos, ROI);

	cout << "ERROR: multiviewDeconvolution<imgType>::readImage :option " << type << " not recognized" << endl;
	return 3;
}

//=======================================================
template<class imgType>
int multiviewDeconvolution<imgType>::allocate_workspace(imgType imgBackground)
{
	//const values throughout the function
	const bool useWeights = (weights.getPointer_CPU(0) != NULL);
	const int64_t nImg = img.numElements(0);
	const size_t nViews = img.getNumberOfViews();
	const int64_t imSizeFFT = nImg + (2 * img.dimsImgVec[0].dims[2] * img.dimsImgVec[0].dims[1]); //size of the R2C transform in cuFFTComple

	//variables needed for this function	
	psfType *psf_notPadded_GPU = NULL;//to store original PSF

	if (nViews == 0)
	{
		cout << "ERROR:multiviewDeconvolution<imgType>::allocate_workspace(): no views loaded to start process" << endl;
		return 2;
	}

	//allocate temporary mmeory to nromalize weights
	weightType *weightAvg_GPU = NULL;
	if (useWeights)
	{
		HANDLE_ERROR(cudaMalloc((void**)&(weightAvg_GPU), nImg * sizeof(weightType)));
		HANDLE_ERROR(cudaMemset(weightAvg_GPU, 0, nImg * sizeof(weightType)));
	}



	//preparing FFT plans
	cufftPlan3d(&fftPlanFwd, img.dimsImgVec[0].dims[2], img.dimsImgVec[0].dims[1], img.dimsImgVec[0].dims[0], CUFFT_R2C); HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd, CUFFT_COMPATIBILITY_NATIVE); HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	cufftPlan3d(&fftPlanInv, img.dimsImgVec[0].dims[2], img.dimsImgVec[0].dims[1], img.dimsImgVec[0].dims[0], CUFFT_C2R); HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanInv, CUFFT_COMPATIBILITY_NATIVE); HANDLE_ERROR_KERNEL;

	//allocate memory and precompute things for each view things for each vieww
	cout << "===================TODO: load img and weights on the fly to CPU to avoid consuming too much memory====================" << endl;
	for (size_t ii = 0; ii < nViews; ii++)
	{
		//load img for ii-th to CPU 
		//cout << "===================TODO: load weights on the fly to CPU to avoid consuming too much memory====================" << endl;
		//allocate memory for image in the GPU		
		img.allocateView_GPU(ii, nImg * sizeof(imgType));
		//transfer image
		HANDLE_ERROR(cudaMemcpy(img.getPointer_GPU(ii), img.getPointer_CPU(ii), nImg * sizeof(imgType), cudaMemcpyHostToDevice));
		//deallocate memory from CPU
		img.deallocateView_CPU(ii);
        //subtract background
		if ( imgBackground > 0)
			elementwiseOperationInPlace<imgType>(img.getPointer_GPU(ii), imgBackground, nImg, op_elementwise_type::minus_positive);

		if (useWeights)
		{
			//load weights for ii-th to CPU 
			//cout << "===================TODO: load weights on the fly to CPU to avoid consuming too much memory====================" << endl;
			//allocate memory for weights in the GPU			
			weights.allocateView_GPU(ii, nImg * sizeof(weightType));
			//transfer image
			HANDLE_ERROR(cudaMemcpy(weights.getPointer_GPU(ii), weights.getPointer_CPU(ii), nImg * sizeof(weightType), cudaMemcpyHostToDevice));
			//deallocate memory from CPU
			weights.deallocateView_CPU(ii);

			//call kernel to update weightAvg_GPU
			elementwiseOperationInPlace<weightType>(weightAvg_GPU, weights.getPointer_GPU(ii), nImg, op_elementwise_type::plus);
		}

		//allocate memory for PSF FFT
		const int64_t psfSize = psf.numElements(ii);
		HANDLE_ERROR(cudaMalloc((void**)&(psf_notPadded_GPU), (psfSize) * sizeof(psfType)));		
		psf.allocateView_GPU(ii, imSizeFFT * sizeof(psfType));

		//transfer psf
		HANDLE_ERROR(cudaMemcpy(psf_notPadded_GPU, psf.getPointer_CPU(ii), psfSize * sizeof(psfType), cudaMemcpyHostToDevice));

		//apply ffshift to kernel and pad it with zeros so we can calculate convolution with FFT
		int numThreads = std::min((int64_t)MAX_THREADS_CUDA/4, psfSize);
		int numBlocks = std::min((int64_t)MAX_BLOCKS_CUDA, (int64_t)(psfSize + (int64_t)(numThreads - 1)) / ((int64_t)numThreads));
		HANDLE_ERROR(cudaMemset(psf.getPointer_GPU(ii), 0, imSizeFFT * sizeof(psfType)));		
		fftShiftKernel << <numBlocks, numThreads >> >(psf_notPadded_GPU, psf.getPointer_GPU(ii), psf.dimsImgVec[ii].dims[2], psf.dimsImgVec[ii].dims[1], psf.dimsImgVec[ii].dims[0], img.dimsImgVec[ii].dims[2], img.dimsImgVec[ii].dims[1], img.dimsImgVec[ii].dims[0]); HANDLE_ERROR_KERNEL;


#ifdef _DEBUG
		//char buffer[256];
		//sprintf(buffer, "E:/temp/deconvolution/PSFpadded_view%d.raw", ii);
		//debug_writeGPUarray(psf.getPointer_GPU(ii), img.dimsImgVec[0], string(buffer));		
#endif

		//execute FFT.  If idata and odata are the same, this method does an in-place transform
		cufftExecR2C(fftPlanFwd, psf.getPointer_GPU(ii), (cufftComplex *)(psf.getPointer_GPU(ii))); HANDLE_ERROR_KERNEL;

		//release memory for PSF
		HANDLE_ERROR(cudaFree(psf_notPadded_GPU));
		psf.deallocateView_CPU(ii);
	}	
	
	
	if (useWeights)
	{
		cout << "======TODO: during normalization check elements with all zero weights====" << endl;
		//normalize weights	
		for (size_t ii = 0; ii < nViews; ii++)
		{
			elementwiseOperationInPlace(weights.getPointer_GPU(ii), weightAvg_GPU, nImg, op_elementwise_type::divide);
		}

		//deallocate temporary memory to nromalize weights
		HANDLE_ERROR(cudaFree(weightAvg_GPU));
		weightAvg_GPU = NULL; 
	}


	//allocate memory for final result
	J.resize(1);
	J.setImgDims(0, img.dimsImgVec[0]);
	J.allocateView_GPU(0, nImg * sizeof(outputType));
	J.allocateView_CPU(0, nImg );

	//initialize final results as weighted average of all views
	HANDLE_ERROR(cudaMemset(J.getPointer_GPU(0), 0, nImg * sizeof(outputType)));
	for (size_t ii = 0; ii < nViews; ii++)
	{
		elementwiseOperationOutOfPlace(J.getPointer_GPU(0), weights.getPointer_GPU(ii), img.getPointer_GPU(ii), nImg, op_elementwise_type::compound_plus);
	}


	return 0;
}
//=======================================================
template<class imgType>
int multiviewDeconvolution<imgType>::allocate_workspace_update_multiGPU(imgType imgBackground, bool useWeights)
{
	//const values throughout the function	
	const int64_t nImg = img.numElements(0);
	const size_t nViews = img.getNumberOfViews();
	const int64_t imSizeFFT = nImg + (2 * img.dimsImgVec[0].dims[2] * img.dimsImgVec[0].dims[1]); //size of the R2C transform in cuFFTComple

	
	if (nViews == 0)
	{
		cout << "ERROR:multiviewDeconvolution<imgType>::allocate_workspace(): no views loaded to start process" << endl;
		return 2;
	}

	//allocate temporary memory to nromalize weights
	weightType *weightAvg_GPU = NULL;
	if (useWeights)
	{
		HANDLE_ERROR(cudaMalloc((void**)&(weightAvg_GPU), nImg * sizeof(weightType)));
		HANDLE_ERROR(cudaMemset(weightAvg_GPU, 0, nImg * sizeof(weightType)));
	}

	
	//allocate memory and precompute things for each view things for each vieww	
	for (size_t ii = 0; ii < nViews; ii++)
	{
		//load img for ii-th to CPU 
		//cout << "===================TODO: load weights on the fly to CPU to avoid consuming too much memory====================" << endl;
		//allocate memory for image in the GPU		
		//img.allocateView_GPU(ii, nImg * sizeof(imgType)); memory has already been allocate in the init phase
		//transfer image
		HANDLE_ERROR(cudaMemcpy(img.getPointer_GPU(ii), img.getPointer_CPU(ii), nImg * sizeof(imgType), cudaMemcpyHostToDevice));
		//deallocate memory from CPU
		img.deallocateView_CPU(ii);
		//subtract background
		if (imgBackground > 0)
			elementwiseOperationInPlace<imgType>(img.getPointer_GPU(ii), imgBackground, nImg, op_elementwise_type::minus_positive);

		if (useWeights)
		{
			//load weights for ii-th to CPU 
			//cout << "===================TODO: load weights on the fly to CPU to avoid consuming too much memory====================" << endl;			
			//transfer image
			HANDLE_ERROR(cudaMemcpy(weights.getPointer_GPU(ii), weights.getPointer_CPU(ii), nImg * sizeof(weightType), cudaMemcpyHostToDevice));
			//deallocate memory from CPU
			weights.deallocateView_CPU(ii);

			//call kernel to update weightAvg_GPU
			elementwiseOperationInPlace<weightType>(weightAvg_GPU, weights.getPointer_GPU(ii), nImg, op_elementwise_type::plus);
		}
		
	}


	if (useWeights)
	{
		cout << "======TODO: during normalization check elements with all zero weights====" << endl;
		//normalize weights	
		for (size_t ii = 0; ii < nViews; ii++)
		{
			elementwiseOperationInPlace(weights.getPointer_GPU(ii), weightAvg_GPU, nImg, op_elementwise_type::divide);
		}

		//deallocate temporary memory to nromalize weights
		HANDLE_ERROR(cudaFree(weightAvg_GPU));
		weightAvg_GPU = NULL;
	}
	
	//initialize final results as weighted average of all views
	HANDLE_ERROR(cudaMemset(J.getPointer_GPU(0), 0, nImg * sizeof(outputType)));
	for (size_t ii = 0; ii < nViews; ii++)
	{
		elementwiseOperationOutOfPlace(J.getPointer_GPU(0), weights.getPointer_GPU(ii), img.getPointer_GPU(ii), nImg, op_elementwise_type::compound_plus);
	}


	return 0;
}
//=======================================================
template<class imgType>
int multiviewDeconvolution<imgType>::allocate_workspace_init_multiGPU(const uint32_t blockDims[MAX_DATA_DIMS], bool useWeights)
{

	//const values throughout the function			
	const size_t nViews = psf.getNumberOfViews();	
	int64_t nImg = 1;
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		nImg *= blockDims[ii];

	const int64_t imSizeFFT = nImg + (2 * blockDims[2] * blockDims[1]); //size of the R2C transform in cuFFTComple

	//variables needed for this function	
	psfType *psf_notPadded_GPU = NULL;//to store original PSF

	if (nViews == 0)
	{
		cout << "ERROR:multiviewDeconvolution<imgType>::allocate_workspace(): no views loaded to start process" << endl;
		return 2;
	}
    

	//preparing FFT plans
	cufftPlan3d(&fftPlanFwd, blockDims[2], blockDims[1], blockDims[0], CUFFT_R2C); HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd, CUFFT_COMPATIBILITY_NATIVE); HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	cufftPlan3d(&fftPlanInv, blockDims[2], blockDims[1], blockDims[0], CUFFT_C2R); HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanInv, CUFFT_COMPATIBILITY_NATIVE); HANDLE_ERROR_KERNEL;


	//allocate memory and precompute things for each view things for each vieww	
	for (size_t ii = 0; ii < nViews; ii++)
	{
		//load img for ii-th to CPU 
		//cout << "===================TODO: load weights on the fly to CPU to avoid consuming too much memory====================" << endl;
		//allocate memory for image in the GPU		
		img.allocateView_GPU(ii, nImg * sizeof(imgType));
        //we do not have anything to upload yet		
		
		if (useWeights)
		{
			//load weights for ii-th to CPU 
			//cout << "===================TODO: load weights on the fly to CPU to avoid consuming too much memory====================" << endl;
			//allocate memory for weights in the GPU			
			weights.allocateView_GPU(ii, nImg * sizeof(weightType));
			
		}

		//allocate memory for PSF FFT
		const int64_t psfSize = psf.numElements(ii);
		HANDLE_ERROR(cudaMalloc((void**)&(psf_notPadded_GPU), (psfSize)* sizeof(psfType)));
		psf.allocateView_GPU(ii, imSizeFFT * sizeof(psfType));

		//transfer psf
		HANDLE_ERROR(cudaMemcpy(psf_notPadded_GPU, psf.getPointer_CPU(ii), psfSize * sizeof(psfType), cudaMemcpyHostToDevice));

		//apply ffshift to kernel and pad it with zeros so we can calculate convolution with FFT
		int numThreads = std::min((int64_t)MAX_THREADS_CUDA / 4, psfSize);
		int numBlocks = std::min((int64_t)MAX_BLOCKS_CUDA, (int64_t)(psfSize + (int64_t)(numThreads - 1)) / ((int64_t)numThreads));
		HANDLE_ERROR(cudaMemset(psf.getPointer_GPU(ii), 0, imSizeFFT * sizeof(psfType)));
		fftShiftKernel << <numBlocks, numThreads >> >(psf_notPadded_GPU, psf.getPointer_GPU(ii), psf.dimsImgVec[ii].dims[2], psf.dimsImgVec[ii].dims[1], psf.dimsImgVec[ii].dims[0], blockDims[2], blockDims[1], blockDims[0]); HANDLE_ERROR_KERNEL;


		//execute FFT.  If idata and odata are the same, this method does an in-place transform
		cufftExecR2C(fftPlanFwd, psf.getPointer_GPU(ii), (cufftComplex *)(psf.getPointer_GPU(ii))); HANDLE_ERROR_KERNEL;

		//release memory for PSF
		HANDLE_ERROR(cudaFree(psf_notPadded_GPU));
		psf.deallocateView_CPU(ii);
	}	

	//allocate memory for final result
	J.resize(1);
	dimsImg aux; 
    aux.ndims = MAX_DATA_DIMS;
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		aux.dims[ii] = blockDims[ii];
	J.setImgDims(0, aux);
	J.allocateView_GPU(0, nImg * sizeof(outputType));
	J.allocateView_CPU(0, nImg);	


	return 0;
}

//=======================================================
template<class imgType>
void multiviewDeconvolution<imgType>::deconvolution_LR_TV(int numIters, float lambdaTV)
{
	const bool useWeights = (weights.getPointer_CPU(0) != NULL);
	const int64_t nImg = img.numElements(0);
	const size_t nViews = img.getNumberOfViews();
	const int64_t imSizeFFT = nImg + (2 * img.dimsImgVec[0].dims[2] * img.dimsImgVec[0].dims[1]); //size of the R2C transform in cuFFTComple

	int numThreads = std::min((std::int64_t)MAX_THREADS_CUDA/4, imSizeFFT / 2);//we are using complex numbers
	int numBlocks = std::min((std::int64_t)MAX_BLOCKS_CUDA, (std::int64_t)(imSizeFFT / 2 + (std::int64_t)(numThreads - 1)) / ((std::int64_t)numThreads));    

	//allocate extra memory required for intermediate calculations
	outputType *J_GPU_FFT, *aux_FFT, *aux_LR;
	outputType *TV_GPU = NULL;
	HANDLE_ERROR(cudaMalloc((void**)&(J_GPU_FFT), imSizeFFT * sizeof(outputType)));//for J FFT
	HANDLE_ERROR(cudaMalloc((void**)&(aux_FFT), imSizeFFT * sizeof(outputType)));//to hold products between FFT
	HANDLE_ERROR(cudaMalloc((void**)&(aux_LR), nImg * sizeof(outputType)));//to hold LR update 

	//loop for each iteration
	for (int iter = 0; iter < numIters; iter++)
	{
		//copy current solution
		elementwiseOperationInPlace(J_GPU_FFT, J.getPointer_GPU(0), nImg, op_elementwise_type::copy);
		//precompute FFT for current solution
		cufftExecR2C(fftPlanFwd, J_GPU_FFT, (cufftComplex *)J_GPU_FFT); HANDLE_ERROR_KERNEL;

		//precalculate TV on J
		if (lambdaTV > 0)
		{
			cout << "==============TODO: calculate total variation==================" << endl;
		}

		//reset update
		HANDLE_ERROR(cudaMemset(aux_LR, 0, nImg * sizeof(outputType)));
		//main loop over the different views
		for (int vv = 0; vv < nViews; vv++)
		{
			//multiply LR currant result and kernel in fourier space (and normalize)
			//NOTE: from CUFFT manual: CUFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set followed by an inverse FFT on the resulting set yields data that is equal to the input scaled by the number of elements.			
			modulateAndNormalize_outOfPlace_kernel << <numBlocks, numThreads >> >((cufftComplex *)(aux_FFT), (cufftComplex *)(J_GPU_FFT), (cufftComplex *)(psf.getPointer_GPU(vv)), imSizeFFT / 2, 1.0f / (float)(nImg));//last parameter is the size of the FFT

			//inverse FFT 
			cufftExecC2R(fftPlanInv, (cufftComplex *)aux_FFT, aux_FFT); HANDLE_ERROR_KERNEL;


#ifdef _DEBUG
			char buffer[256];
			sprintf(buffer, "E:/temp/deconvolution/J_iter%.4d.raw", iter);
			if (vv == 0)
				debug_writeGPUarray(J.getPointer_GPU(0), J.dimsImgVec[0], string(buffer));
			/*            
			sprintf(buffer, "E:/temp/deconvolution/img_view%.4d.raw", vv);
			if ( iter == 0 )
				debug_writeGPUarray(img.getPointer_GPU(0), img.dimsImgVec[0], string(buffer));
			sprintf(buffer, "E:/temp/deconvolution/weights_view%.4d.raw", vv);
			if (iter == 0)
				debug_writeGPUarray(weights.getPointer_GPU(0), img.dimsImgVec[0], string(buffer));			
			sprintf(buffer, "E:/temp/deconvolution/JconvPSF_iter%.4d_view%d.raw", iter, vv);
			debug_writeGPUarray(aux_FFT, J.dimsImgVec[0], string(buffer));
			sprintf(buffer, "E:/temp/deconvolution/JFFT_iter%.4d.raw", iter);
			debug_writeGPUarray(J_GPU_FFT, J.dimsImgVec[0], string(buffer));
			sprintf(buffer, "E:/temp/deconvolution/PSFpaddedFfft_iter%.4d_view%d.raw", iter, vv);
			debug_writeGPUarray(psf.getPointer_GPU(vv), J.dimsImgVec[0], string(buffer));			
            */
#endif

			//calculate ratio img.getPointer_GPU(ii) ./ aux_FFT
			elementwiseOperationInPlace(aux_FFT, img.getPointer_GPU(vv), nImg, op_elementwise_type::divide_inv);

			//calculate FFT of ratio (for convolution)
			cufftExecR2C(fftPlanFwd, aux_FFT, (cufftComplex *)aux_FFT); HANDLE_ERROR_KERNEL;

			//multiply auxFFT and FFT(PSF)*
			modulateAndNormalize_conj_kernel << <numBlocks, numThreads >> >((cufftComplex *)(aux_FFT), (cufftComplex *)(psf.getPointer_GPU(vv)), imSizeFFT / 2, 1.0f / (float)(nImg));

			//inverse FFT
			cufftExecC2R(fftPlanInv, (cufftComplex *)aux_FFT, aux_FFT); HANDLE_ERROR_KERNEL;

			//add the value
			if (useWeights)
			{
				elementwiseOperationOutOfPlace(aux_LR, weights.getPointer_GPU(vv), aux_FFT, nImg, op_elementwise_type::compound_multiply);
			}
			else{
				elementwiseOperationInPlace(aux_LR, aux_FFT, nImg, op_elementwise_type::plus);
			}
			
		}

		//normalize weights if we are just using averaging
		if (!useWeights)
			elementwiseOperationInPlace(aux_LR, 1.0f / (float)nViews, nImg, op_elementwise_type::multiply);		

		//apply TV
		if (lambdaTV > 0)
		{
			elementwiseOperationInPlace(aux_LR, TV_GPU, nImg, op_elementwise_type::divide);
		}

		//update LR 
		elementwiseOperationInPlace(J.getPointer_GPU(0), aux_LR, nImg, op_elementwise_type::multiply);
		
	}

	//release memory
	HANDLE_ERROR(cudaFree(aux_LR));
	HANDLE_ERROR(cudaFree(aux_FFT));
	HANDLE_ERROR(cudaFree(J_GPU_FFT));
	if ( TV_GPU != NULL)
		HANDLE_ERROR(cudaFree(TV_GPU));
	
}

//===========================================================
template<class imgType>
void multiviewDeconvolution<imgType>::debug_writDeconvolutionResultRaw(const std::string& filename)
{
	cout << "======DEBUGGING:multiviewDeconvolution<imgType>::debug_writDeconvolutionResultRaw========" << endl;
	cout << " writing raw file " << J.dimsImgVec[0].dims[0] << "x" << J.dimsImgVec[0].dims[1] << "x" << J.dimsImgVec[0].dims[2] << "x" << " in float to " << filename << endl;

	ofstream fid(filename.c_str(), ios::binary);

	fid.write((char*)(J.getPointer_CPU(0)), J.numElements(0) * sizeof(imgType)); 
	fid.close();
}

//===========================================================
template<class imgType>
void multiviewDeconvolution<imgType>::debug_writeGPUarray(float* ptr_GPU, dimsImg& dims, const std::string& filename)
{
	cout << "======DEBUGGING:multiviewDeconvolution<imgType>::debug_writeGPUarray========" << endl;
	cout << " writing raw file ";
	int64_t numElements = 1;

	for (int ii = 0; ii < dims.ndims; ii++)
	{
		numElements *= dims.dims[ii];
		cout << dims.dims[ii] << "x";
	}
	cout << " in float format to " << filename << endl;


	float* ptr_CPU = new float[numElements];
	HANDLE_ERROR(cudaMemcpy(ptr_CPU, ptr_GPU, numElements * sizeof(float), cudaMemcpyDeviceToHost));

	ofstream fid(filename.c_str(), ios::binary);
	fid.write((char*)(ptr_CPU), numElements * sizeof(float));
	fid.close();

	delete[] ptr_CPU;
}

//===========================================================
template<class imgType>
void multiviewDeconvolution<imgType>::debug_writeCPUarray(float* ptr_CPU, dimsImg& dims, const std::string& filename)
{
	cout << "======DEBUGGING:multiviewDeconvolution<imgType>::debug_writeCPUarray========" << endl;
	cout << " writing raw file ";
	int64_t numElements = 1;

	for (int ii = 0; ii < dims.ndims; ii++)
	{
		numElements *= dims.dims[ii];
		cout << dims.dims[ii] << "x";
	}
	cout << " in float format to " << filename << endl;


	
	ofstream fid(filename.c_str(), ios::binary);
	fid.write((char*)(ptr_CPU), numElements * sizeof(float));
	fid.close();

	
}


//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
//NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting factor 

//uncomment to save intermediate steps
#define DEBUG_FFT_INTERMEDIATE_STEPS

template<class imgType>
imgType* multiviewDeconvolution<imgType>::convolution3DfftCUDA(const imgType* im, const std::int64_t* imDim, const imgType* kernel, const std::int64_t* kernelDim, int devCUDA)
{
	imgType* convResult = NULL;
	imgType* imCUDA = NULL;
	imgType* kernelCUDA = NULL;
	imgType* kernelPaddedCUDA = NULL;

	int dimsImage = 3;

	cufftHandle fftPlanFwd, fftPlanInv;

#ifdef DEBUG_FFT_INTERMEDIATE_STEPS
	string filepath("E:/temp/deconvolution/");
#endif

	HANDLE_ERROR(cudaSetDevice(devCUDA));

	long long int imSize = 1;
	long long int kernelSize = 1;
	for (int ii = 0; ii<dimsImage; ii++)
	{
		imSize *= (long long int) (imDim[ii]);
		kernelSize *= (long long int) (kernelDim[ii]);
	}

	long long int imSizeFFT = imSize + (long long int)(2 * imDim[0] * imDim[1]); //size of the R2C transform in cuFFTComplex

	//allocate memory for output result
	convResult = new imgType[imSize];

	//allocat ememory in GPU
	HANDLE_ERROR(cudaMalloc((void**)&(imCUDA), imSizeFFT*sizeof(imgType)));//a little bit larger to allow in-place FFT
	HANDLE_ERROR(cudaMalloc((void**)&(kernelCUDA), (kernelSize)*sizeof(imgType)));
	HANDLE_ERROR(cudaMalloc((void**)&(kernelPaddedCUDA), imSizeFFT*sizeof(imgType)));


	//TODO: pad image to a power of 2 size in all dimensions (use whatever  boundary conditions you want to apply)
	//TODO: pad kernel to image size
	//TODO: pad kernel and image to xy(z/2 + 1) for in-place transform
	//NOTE: in the example for 2D convolution using FFT in the Nvidia SDK they do the padding in the GPU, but in might be pushing the memory in the GPU for large images.

	//printf("Copying memory (kernel and image) to GPU\n");
	HANDLE_ERROR(cudaMemcpy(kernelCUDA, kernel, kernelSize*sizeof(imgType), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(imCUDA, im, imSize*sizeof(imgType), cudaMemcpyHostToDevice));

	//apply ffshift to kernel and pad it with zeros so we can calculate convolution with FFT
	HANDLE_ERROR(cudaMemset(kernelPaddedCUDA, 0, imSizeFFT*sizeof(imgType)));
	int numThreads = std::min((long long int)MAX_THREADS_CUDA, kernelSize);
	int numBlocks = std::min((long long int)MAX_BLOCKS_CUDA, (long long int)(kernelSize + (long long int)(numThreads - 1)) / ((long long int)numThreads));
	fftShiftKernel << <numBlocks, numThreads >> >(kernelCUDA, kernelPaddedCUDA, kernelDim[0], kernelDim[1], kernelDim[2], imDim[0], imDim[1], imDim[2]); HANDLE_ERROR_KERNEL;


	//make sure GPU finishes before we launch two different streams
	HANDLE_ERROR(cudaDeviceSynchronize());

#ifdef DEBUG_FFT_INTERMEDIATE_STEPS
	dimsImg auxDimsImg;
	auxDimsImg.ndims = dimsImage;
	for (int ii = 0; ii < dimsImage; ii++)
		auxDimsImg.dims[ii] = imDim[dimsImage-1-ii];//flip coordinates

	debug_writeGPUarray(kernelPaddedCUDA, auxDimsImg, string(filepath + "cudafft3d_kernelPaddedCuda.raw"));
#endif

	//printf("Creating R2C & C2R FFT plans for size %i x %i x %i\n",imDim[0],imDim[1],imDim[2]);
	cufftPlan3d(&fftPlanFwd, imDim[0], imDim[1], imDim[2], CUFFT_R2C); HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd, CUFFT_COMPATIBILITY_NATIVE); HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R); HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanInv, CUFFT_COMPATIBILITY_NATIVE); HANDLE_ERROR_KERNEL;

	//transforming convolution kernel; TODO: if I do multiple convolutions with the same kernel I could reuse the results at teh expense of using out-of place memory (and then teh layout of the data is different!!!! so imCUDAfft should also be out of place)
	//NOTE: from CUFFT manual: If idata and odata are the same, this method does an in-place transform.
	//NOTE: from CUFFT manual: inplace output data xy(z/2 + 1) with fcomplex. Therefore, in order to perform an in-place FFT, the user has to pad the input array in the last dimension to Nn2 + 1 complex elements interleaved. Note that the real-to-complex transform is implicitly forward.
	cufftExecR2C(fftPlanFwd, imCUDA, (cufftComplex *)imCUDA); HANDLE_ERROR_KERNEL;
	//transforming image
	cufftExecR2C(fftPlanFwd, kernelPaddedCUDA, (cufftComplex *)kernelPaddedCUDA); HANDLE_ERROR_KERNEL;

#ifdef DEBUG_FFT_INTERMEDIATE_STEPS
	debug_writeGPUarray(kernelPaddedCUDA, auxDimsImg, string(filepath + "cudafft3d_kernelPaddedCuda_fft.raw"));
	debug_writeGPUarray(imCUDA, auxDimsImg, string(filepath + "cudafft3d_im_fft.raw"));
#endif


	//multiply image and kernel in fourier space (and normalize)
	//NOTE: from CUFFT manual: CUFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set followed by an inverse FFT on the resulting set yields data that is equal to the input scaled by the number of elements.
	numThreads = std::min((long long int)MAX_THREADS_CUDA, imSizeFFT / 2);//we are using complex numbers
	numBlocks = std::min((long long int)MAX_BLOCKS_CUDA, (long long int)(imSizeFFT / 2 + (long long int)(numThreads - 1)) / ((long long int)numThreads));
	modulateAndNormalize_kernel << <numBlocks, numThreads >> >((cufftComplex *)imCUDA, (cufftComplex *)kernelPaddedCUDA, imSizeFFT / 2, 1.0f / (float)(imSize));//last parameter is the size of the FFT

#ifdef DEBUG_FFT_INTERMEDIATE_STEPS	
	debug_writeGPUarray(imCUDA, auxDimsImg, string(filepath + "cudafft3d_imTimesPSF_fft.raw"));
#endif

	//inverse FFT 
	cufftExecC2R(fftPlanInv, (cufftComplex *)imCUDA, imCUDA); HANDLE_ERROR_KERNEL;

	//copy result to host
	HANDLE_ERROR(cudaMemcpy(convResult, imCUDA, sizeof(imgType)*imSize, cudaMemcpyDeviceToHost));

	//release memory
	(cufftDestroy(fftPlanInv)); HANDLE_ERROR_KERNEL;
	(cufftDestroy(fftPlanFwd)); HANDLE_ERROR_KERNEL;
	HANDLE_ERROR(cudaFree(imCUDA));
	HANDLE_ERROR(cudaFree(kernelCUDA));
	HANDLE_ERROR(cudaFree(kernelPaddedCUDA));

	return convResult;
}

//=================================================================
template<class imgType>
imgType* multiviewDeconvolution<imgType>::convolution3DfftCUDA_img_psf(size_t pos, int devCUDA)
{ 
	//we need to flip dimensions because cuFFT the fastest running dimension is the last one
	int64_t dimsI[3], dimsP[3];

	dimsI[0] = img.dimsImgVec[pos].dims[2];
	dimsI[1] = img.dimsImgVec[pos].dims[1];
	dimsI[2] = img.dimsImgVec[pos].dims[0];

	dimsP[0] = psf.dimsImgVec[pos].dims[2];
	dimsP[1] = psf.dimsImgVec[pos].dims[1];
	dimsP[2] = psf.dimsImgVec[pos].dims[0];

	imgType* aux = convolution3DfftCUDA(img.getPointer_CPU(pos), dimsI, psf.getPointer_CPU(pos), dimsP, devCUDA);
	//imgType* aux = convolution3DfftCUDA(img.getPointer_CPU(pos), img.dimsImgVec[pos].dims, psf.getPointer_CPU(pos), psf.dimsImgVec[pos].dims, devCUDA); 

	return aux;
};

//=================================================================
//declare all possible instantitation for the template
//TODO: right now the code can only handle float images since the rest of operations are carried in float point
//template class multiviewDeconvolution<uint16_t>;
//template class multiviewDeconvolution<uint8_t>;
template class multiviewDeconvolution<float>;