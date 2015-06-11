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



/**
*	functor for adding two numbers
*/
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
		break;HANDLE_ERROR_KERNEL;

	case op_elementwise_type::divide:
		elementwiseOperationInPlace_kernel << <numBlocks, numThreads >> > (A, B, arrayLength, div_func<T>()); HANDLE_ERROR_KERNEL;
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
