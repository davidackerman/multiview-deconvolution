/*
* Copyright (C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  commonCUDA.h
*
*  Created on: June 5th, 2015
*      Author: Fernando Amat
*
* \brief common functions and constants for CUDA
*/

#ifndef __FA_COMMON_CUDA_H__
#define __FA_COMMON_CUDA_H__

#include <cstdint>

#ifndef FA_CUDA_CONSTANTS_
#define FA_CUDA_CONSTANTS_
//define constants. We do not support cuda compute capability < 2.0
#if __CUDA_ARCH__ < 300
static const int MAX_THREADS_CUDA = 1024; //maximum number of threads per block
static const int MAX_BLOCKS_CUDA = 65535;
#else
static const int MAX_THREADS_CUDA = 1024; //maximum number of threads per block
static const int MAX_BLOCKS_CUDA = 2147483647;
#endif

#endif

//defines the types of operations implement for elementwise function
enum op_elementwise_type { plus, minus, multiply, divide, divide_inv, compound_plus, copy, compound_multiply, minus_positive, isnanop};



/*
\brief A[i] = f(A[i],B[i])  where f is defined by the enum op
*/
template<class T>
void elementwiseOperationInPlace(T* A, const T* B, std::uint64_t arrayLength, op_elementwise_type op);

/*
\brief A[i] = f(A[i],B)  where f is defined by the enum op
*/
template<class T>
void elementwiseOperationInPlace(T* A, const T B, std::uint64_t arrayLength, op_elementwise_type op);

/*
\brief C[i] = f(A[i],B[i])  where f is defined by the enum op
*/
template<class T>
void elementwiseOperationOutOfPlace(T* C, const T* A, const T* B, std::uint64_t arrayLength, op_elementwise_type op);



void elementwiseOperationInPlace_TVreg(float* A, const float* B, std::uint64_t arrayLength, float lambdaTV);


//=============================================================================
//reduction operations
//==============================================================================
enum op_reduction_type {add, min_elem, max_elem};

/*
return max(A) for example or sum(A)
*/
template<class T>
T reductionOperation(const T* A, std::uint64_t arrayLength, op_reduction_type op);



//=============================================================================
//memory allocation and deallocation
//==============================================================================


template<class T>
T* allocateMem_GPU(size_t numElem);

template<class T>
void deallocateMem_GPU(T* ptr);

template<class T>
void copy_GPU_to_CPU(T* ptr_CPU, const T* ptr_CUDA, size_t numElem);

template<class T>
void copy_CPU_to_GPU(const T* ptr_CPU, T* ptr_CUDA, size_t numElem);

#endif