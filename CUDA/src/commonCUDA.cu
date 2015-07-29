/*
* Copyright (C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  commonCUDA.cu
*
*  Created on: June 5th, 2015
*      Author: Fernando Amat
*
* \brief common functions and constants for CUDA
*/


#include <cstdint>
#include <algorithm>
#include <iostream>
#include "commonCUDA.h"
#include "cuda.h"
#include "book.h"

using namespace std;

//profiling it is better to not use all the threads
#define THREADS_CUDA_FOR_REDUCTION (MAX_THREADS_CUDA / 4)


/**
*	functor for adding two numbers
*/
template <class T>
struct max_func
{
	max_func(){};
	__device__ T operator () (const T& a, const T& b) { return (a > b ? a : b); }
};

template <class T>
struct min_func
{
	min_func(){};
	__device__ T operator () (const T& a, const T& b) { return (a > b ? b : a); }
};

template <class T>
struct add_func
{
	add_func(){};
	__device__ T operator () (const T& a, const T& b) { return a + b; }
};

template <class T>
struct sub_func
{
	sub_func(){};
	__device__ T operator () (const T& a, const T& b) { return a - b; }
};

template <class T>
struct sub_pos_func
{
	sub_pos_func(){};
	__device__ T operator () (const T& a, const T& b) { return (a > b ? a-b : 0); }//lower bounded by zero
};

template <class T>
struct div_func
{
	div_func(){};
	__device__ T operator () (const T& a, const T& b) { return a / b; }
};

template <class T>
struct div_inv_func
{
	div_inv_func(){};
	__device__ T operator () (const T& a, const T& b) { return b / a; }
};

template <class T>
struct mul_func
{
	mul_func(){};
	__device__ T operator () (const T& a, const T& b) { return a * b; }
};

template <class T>
struct isnan_func
{
	isnan_func(){};
	__device__ T operator () (const T& a, const T& b) { return a; }
};

template <>
struct isnan_func <float>
{
	isnan_func(){};
	__device__ float operator () (const float& a, const float& b) { return(::isnan(a) ? b : a); }
};

template <>
struct isnan_func <double>
{
	isnan_func(){};
	__device__ double operator () (const double& a, const double& b) { return(::isnan(a) ? b : a); }
};

template <class T>
struct equal_func
{
	equal_func(){};
	__device__ T operator () (const T& a, const T& b) { return b; }
};


//inspired by https://github.com/DrMikeMorgan/Cuda/blob/master/functors.cu.h on how to use functors for CUDA
//starting with CUDA 7.0 we can probably use lambda functions instead of struct (CUDA 7.0 inciroirates C++11 standards)
template<class T, class operation>
__global__ void elementwiseOperationInPlace_kernel(T *A, const T *B, std::uint64_t arrayLength, operation op)
{
	std::uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (tid < arrayLength)
	{
		A[tid] = op(A[tid], B[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

//==============================================================================================
template<class T>
__global__ void elementwiseOperationInPlace__TVreg_kernel(T *A, const T *B, std::uint64_t arrayLength, T lambda)
{
	std::uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (tid < arrayLength)
	{		
		A[tid] /= max(1.0 - lambda * B[tid], 1e-3);//we avoid "crazy" updates by setting a maximum
		tid += blockDim.x * gridDim.x;
	}
}
//==============================================================================================
template<class T, class operation>
__global__ void elementwiseOperationInPlace_kernel(T *A, const T B, std::uint64_t arrayLength, operation op)
{
	std::uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (tid < arrayLength)
	{
		A[tid] = op(A[tid], B);
		tid += blockDim.x * gridDim.x;
	}
}

//==============================================================================================
template<class T, class operation>
__global__ void elementwiseOperationOutOfPlace_kernel(T* C, const T *A, const T *B, std::uint64_t arrayLength, operation op)
{
	std::uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (tid < arrayLength)
	{
		C[tid] = op(A[tid], B[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

//==============================================================================================
template<class T, class operation>
__global__ void elementwiseOperationOutOfPlace_compund_kernel(T* C, const T *A, const T *B, std::uint64_t arrayLength, operation op)
{
	std::uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (tid < arrayLength)
	{
		C[tid] += op(A[tid], B[tid]);
		tid += blockDim.x * gridDim.x;
	}
}


//==============================================================================================
template<class T, class operation>
__global__ void reductionOperation_kernel(const T *A, T* temp_accumulator_CUDA, std::uint64_t arrayLength, operation op, T defaultVal)
{

	__shared__ T copyShared[THREADS_CUDA_FOR_REDUCTION];//blockDim.x = THREADS_CUDA_FOR_REDUCTION
	

	//copy to share memory
	if (blockDim.x * blockIdx.x + threadIdx.x < arrayLength)
		copyShared[threadIdx.x] = A[blockDim.x * blockIdx.x + threadIdx.x];
	else
		copyShared[threadIdx.x] = defaultVal;//depending on the reduction operation we want different default values here
	__syncthreads();

	
	//perform reduction
	int i = blockDim.x / 2;	
	while (i != 0)
	{
		if (threadIdx.x < i)
			copyShared[threadIdx.x] = op(copyShared[threadIdx.x], copyShared[threadIdx.x + i]);
		__syncthreads();
		i /= 2;
	}

	//store reduction value for this block
	if ( threadIdx.x == 0)
		temp_accumulator_CUDA[blockIdx.x] = copyShared[0];

}

//==============================================================================================
void elementwiseOperationInPlace_TVreg(float* A, const float* B, std::uint64_t arrayLength, float lambdaTV)
{
	int numThreads = std::min((uint64_t)MAX_THREADS_CUDA / 4, arrayLength);//profiling it is better to not use all threads for better occupancy
	int numBlocks = std::min((uint64_t)MAX_BLOCKS_CUDA, (uint64_t)(arrayLength + (uint64_t)(numThreads - 1)) / ((uint64_t)numThreads));

	elementwiseOperationInPlace__TVreg_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, lambdaTV); HANDLE_ERROR_KERNEL;
}

//==============================================================================================
template<class T>
void elementwiseOperationInPlace(T* A, const T* B, std::uint64_t arrayLength, op_elementwise_type op)
{

	int numThreads = std::min((uint64_t)MAX_THREADS_CUDA / 4, arrayLength);//profiling it is better to not use all threads for better occupancy
	int numBlocks = std::min((uint64_t)MAX_BLOCKS_CUDA, (uint64_t)(arrayLength + (uint64_t)(numThreads - 1)) / ((uint64_t)numThreads));


	switch (op)
	{
	case op_elementwise_type::plus:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, add_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::minus:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, sub_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::multiply:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, mul_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::divide:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, div_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	case op_elementwise_type::divide_inv:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, div_inv_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	case op_elementwise_type::copy:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, equal_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	case op_elementwise_type::minus_positive:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, sub_pos_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	default:
		cout << "ERROR: elementwiseOperationInPlace: operation not supported" << endl;
	}
	
}

//==============================================================================================
template<class T>
void elementwiseOperationInPlace(T* A, const T B, std::uint64_t arrayLength, op_elementwise_type op)
{

	int numThreads = std::min((uint64_t)MAX_THREADS_CUDA / 4, arrayLength);//profiling it is better to not use all threads for better occupancy
	int numBlocks = std::min((uint64_t)MAX_BLOCKS_CUDA, (uint64_t)(arrayLength + (uint64_t)(numThreads - 1)) / ((uint64_t)numThreads));


	switch (op)
	{
	case op_elementwise_type::plus:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, add_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::minus:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, sub_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::multiply:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, mul_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::divide:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, div_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	case op_elementwise_type::copy:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, equal_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::minus_positive:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, sub_pos_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	case op_elementwise_type::isnanop:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, isnan_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	default:
		cout << "ERROR: elementwiseOperationInPlace: operation not supported" << endl;
	}

}


//==========================================================================================================
template<class T>
void elementwiseOperationOutOfPlace(T* C, const T* A, const T* B, std::uint64_t arrayLength, op_elementwise_type op)
{

	int numThreads = std::min((uint64_t)MAX_THREADS_CUDA / 4, arrayLength);//profiling it is better to not use all threads for better occupancy
	int numBlocks = std::min((uint64_t)MAX_BLOCKS_CUDA, (uint64_t)(arrayLength + (uint64_t)(numThreads - 1)) / ((uint64_t)numThreads));


	switch (op)
	{
	case op_elementwise_type::plus:
		elementwiseOperationOutOfPlace_kernel << <numBlocks, numThreads >> > (C, A, B, arrayLength, add_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::minus:
		elementwiseOperationOutOfPlace_kernel << <numBlocks, numThreads >> > (C, A, B, arrayLength, sub_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::multiply:
		elementwiseOperationOutOfPlace_kernel << <numBlocks, numThreads >> > (C, A, B, arrayLength, mul_func<T>()); HANDLE_ERROR_KERNEL;
		break;

	case op_elementwise_type::divide:
		elementwiseOperationOutOfPlace_kernel << <numBlocks, numThreads >> > (C, A, B, arrayLength, div_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	case op_elementwise_type::compound_plus:
		elementwiseOperationOutOfPlace_compund_kernel << <numBlocks, numThreads >> > (C, A, B, arrayLength, add_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	case op_elementwise_type::compound_multiply:
		elementwiseOperationOutOfPlace_compund_kernel << <numBlocks, numThreads >> > (C, A, B, arrayLength, mul_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	case op_elementwise_type::minus_positive:
		elementwiseOperationOutOfPlace_kernel << <numBlocks, numThreads >> > (C, A, B, arrayLength, sub_pos_func<T>()); HANDLE_ERROR_KERNEL;
		break;
	default:
		cout << "ERROR: elementwiseOperationInPlace: operation not supported" << endl;
	}

}

//======================================================================
template<class T>
T reductionOperation(const T* A, std::uint64_t arrayLength, op_reduction_type op)
{
	const int numThreads = THREADS_CUDA_FOR_REDUCTION;//profiling it is better to not use all threads for better occupancy
	const int numBlocks = std::min((uint64_t)MAX_BLOCKS_CUDA, (uint64_t)(arrayLength + (uint64_t)(numThreads - 1)) / ((uint64_t)numThreads));

	const uint64_t chunkSize = ((uint64_t)(numThreads)) * ((uint64_t)(numBlocks));

	//allocate temporary memory to finish the reduction on the CPU
	T* reduction_CPU;
	T* reduction_GPU;
	HANDLE_ERROR(cudaMalloc((void**)&(reduction_GPU), numBlocks * sizeof(T)));
	HANDLE_ERROR(cudaMallocHost((void**)&reduction_CPU, numBlocks * sizeof(T))); // host pinned for faster transfers

	//initialize result
	T finalVal;
	switch (op)
	{
	case op_reduction_type::add:
		finalVal = 0;
		break;
	case op_reduction_type::max_elem:
		finalVal = numeric_limits<T>::min();
		break; 
	case op_reduction_type::min_elem:
		finalVal = numeric_limits<T>::max();
		break;
	default:
		cout << "ERROR: reductionOperation: operation not supported" << endl;
	}

	//main loop
	std::uint64_t offset = 0;
	std::uint64_t length, arrayLengthOrig = arrayLength;
	while (offset < arrayLengthOrig)
	{
		const T* ptr = &(A[offset]);
		length = min(arrayLength, chunkSize);

		switch (op)
		{
		case op_reduction_type::add:
			reductionOperation_kernel << <numBlocks, numThreads >> >(ptr, reduction_GPU, length, add_func<T>(), T(0)); HANDLE_ERROR_KERNEL;
			HANDLE_ERROR(cudaMemcpy(reduction_CPU, reduction_GPU, numBlocks * sizeof(T), cudaMemcpyDeviceToHost));
			for (int ii = 0; ii < numBlocks; ii++)
				finalVal += reduction_CPU[ii];

			break;
		case op_reduction_type::max_elem:
			reductionOperation_kernel << <numBlocks, numThreads >> >(ptr, reduction_GPU, length, max_func<T>(), numeric_limits<T>::min()); HANDLE_ERROR_KERNEL;
			HANDLE_ERROR(cudaMemcpy(reduction_CPU, reduction_GPU, numBlocks * sizeof(T), cudaMemcpyDeviceToHost));
			for (int ii = 0; ii < numBlocks; ii++)
				finalVal = max(reduction_CPU[ii], finalVal);

			break;
		case op_reduction_type::min_elem:
			reductionOperation_kernel << <numBlocks, numThreads >> >(ptr, reduction_GPU, length, min_func<T>(), numeric_limits<T>::max()); HANDLE_ERROR_KERNEL;
			HANDLE_ERROR(cudaMemcpy(reduction_CPU, reduction_GPU, numBlocks * sizeof(T), cudaMemcpyDeviceToHost));
			for (int ii = 0; ii < numBlocks; ii++)
				finalVal = min(reduction_CPU[ii], finalVal);

			break;
		default:
			cout << "ERROR: reductionOperation: operation not supported" << endl;
		}


		offset += chunkSize;
		arrayLength -= chunkSize;
	}

	//release memory	
	HANDLE_ERROR(cudaFree(reduction_GPU));
	HANDLE_ERROR(cudaFreeHost(reduction_CPU));


	return finalVal;
}

//======================================================================
//instantiate templates
template void elementwiseOperationInPlace<std::uint8_t>(std::uint8_t* A, const std::uint8_t* B, std::uint64_t arrayLength, op_elementwise_type op);
template void elementwiseOperationInPlace<std::uint16_t>(std::uint16_t* A, const std::uint16_t* B, std::uint64_t arrayLength, op_elementwise_type op);
template void elementwiseOperationInPlace<float>(float* A, const float* B, std::uint64_t arrayLength, op_elementwise_type op);


template void elementwiseOperationInPlace<std::uint8_t>(std::uint8_t* A, const std::uint8_t B, std::uint64_t arrayLength, op_elementwise_type op);
template void elementwiseOperationInPlace<std::uint16_t>(std::uint16_t* A, const std::uint16_t B, std::uint64_t arrayLength, op_elementwise_type op);
template void elementwiseOperationInPlace<float>(float* A, const float B, std::uint64_t arrayLength, op_elementwise_type op);

template void elementwiseOperationOutOfPlace<float>(float* C, const float* A, const float* B, std::uint64_t arrayLength, op_elementwise_type op);
template void elementwiseOperationOutOfPlace<std::uint16_t>(std::uint16_t* C, const std::uint16_t* A, const std::uint16_t* B, std::uint64_t arrayLength, op_elementwise_type op);
template void elementwiseOperationOutOfPlace<std::uint8_t>(std::uint8_t* C, const std::uint8_t* A, const std::uint8_t* B, std::uint64_t arrayLength, op_elementwise_type op);


template float reductionOperation<float>(const float* A, std::uint64_t arrayLength, op_reduction_type op);
template std::uint8_t reductionOperation<std::uint8_t>(const std::uint8_t* A, std::uint64_t arrayLength, op_reduction_type op);
template std::uint16_t reductionOperation<std::uint16_t>(const std::uint16_t* A, std::uint64_t arrayLength, op_reduction_type op);
template double reductionOperation<double>(const double* A, std::uint64_t arrayLength, op_reduction_type op);