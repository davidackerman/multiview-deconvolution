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


//===========================================================================

template<class imgType>
multiviewDeconvolution<imgType>::multiviewDeconvolution()
{
	J.resize(1);//allocate for the output	
}

//=======================================================
template<class imgType>
multiviewDeconvolution<imgType>::~multiviewDeconvolution()
{
	cout << "============TODO: multiviewDeconvolution<imgType>::~multiviewDeconvolution() verify typical values of cufftHandle to add a check before destroying them==============" << endl;
	(cufftDestroy(fftPlanInv)); HANDLE_ERROR_KERNEL;
	(cufftDestroy(fftPlanFwd)); HANDLE_ERROR_KERNEL;
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
int multiviewDeconvolution<imgType>::allocate_workspace()
{
	//const values throughout the function
	const bool useWeights = (weights.getPointer_CPU(0) != NULL);
	const int64_t nImg = img.numElements(0);
	const size_t nViews = img.getNumberOfViews();
	const int64_t imSizeFFT = nImg + (2 * img.dimsImgVec[0].dims[0] * img.dimsImgVec[0].dims[1]); //size of the R2C transform in cuFFTComple

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
	cufftPlan3d(&fftPlanFwd, img.dimsImgVec[0].dims[0], img.dimsImgVec[0].dims[1], img.dimsImgVec[0].dims[2], CUFFT_R2C); HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd, CUFFT_COMPATIBILITY_NATIVE); HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	cufftPlan3d(&fftPlanInv, img.dimsImgVec[0].dims[0], img.dimsImgVec[0].dims[1], img.dimsImgVec[0].dims[2], CUFFT_C2R); HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanInv, CUFFT_COMPATIBILITY_NATIVE); HANDLE_ERROR_KERNEL;

	//allocate memory and precompute things for each view things for each vieww
	for (size_t ii = 0; ii < nViews; ii++)
	{
		//load img for ii-th to CPU 
		cout << "===================TODO: load weights on the fly to CPU to avoid consuming too much memory====================" << endl;
		//allocate memory for image in the GPU		
		img.allocateView_GPU(ii, nImg * sizeof(imgType));
		//transfer image
		HANDLE_ERROR(cudaMemcpy(img.getPointer_GPU(ii), img.getPointer_CPU(ii), nImg * sizeof(imgType), cudaMemcpyHostToDevice));
		//deallocate memory from CPU
		img.deallocateView_CPU(ii);

		if (useWeights)
		{
			//load weights for ii-th to CPU 
			cout << "===================TODO: load weights on the fly to CPU to avoid consuming too much memory====================" << endl;
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
		int numThreads = std::min((int64_t)MAX_THREADS_CUDA, psfSize);
		int numBlocks = std::min((int64_t)MAX_BLOCKS_CUDA, (int64_t)(psfSize + (int64_t)(numThreads - 1)) / ((int64_t)numThreads));
		HANDLE_ERROR(cudaMemset(psf.getPointer_GPU(ii), 0, imSizeFFT * sizeof(psfType)));		
		fftShiftKernel << <numBlocks, numThreads >> >(psf_notPadded_GPU, psf.getPointer_GPU(ii), psf.dimsImgVec[ii].dims[0], psf.dimsImgVec[ii].dims[1], psf.dimsImgVec[ii].dims[2], img.dimsImgVec[ii].dims[0], img.dimsImgVec[ii].dims[1], img.dimsImgVec[ii].dims[2]); HANDLE_ERROR_KERNEL;

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
void multiviewDeconvolution<imgType>::deconvolution_LR_TV(int numIters, float lambdaTV)
{
	cout << "===================TODO=================" << endl;
}


//=================================================================
//declare all possible instantitation for the template
//TODO: right now the code can only handle float images since the rest of operations are carried in float point
//template class multiviewDeconvolution<uint16_t>;
//template class multiviewDeconvolution<uint8_t>;
template class multiviewDeconvolution<float>;