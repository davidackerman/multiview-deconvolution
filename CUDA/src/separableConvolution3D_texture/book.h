/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __BOOK_H__
#define __BOOK_H__
#include <stdio.h>
#include "cufft.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

/*
typedef enum cufftResult_t {
	CUFFT_SUCCESS = 0,  //  The cuFFT operation was successful
	CUFFT_INVALID_PLAN = 1,  //  cuFFT was passed an invalid plan handle
	CUFFT_ALLOC_FAILED = 2,  //  cuFFT failed to allocate GPU or CPU memory
	CUFFT_INVALID_TYPE = 3,  //  No longer used
	CUFFT_INVALID_VALUE = 4,  //  User specified an invalid pointer or parameter
	CUFFT_INTERNAL_ERROR = 5,  //  Driver or internal cuFFT library error
	CUFFT_EXEC_FAILED = 6,  //  Failed to execute an FFT on the GPU
	CUFFT_SETUP_FAILED = 7,  //  The cuFFT library failed to initialize
	CUFFT_INVALID_SIZE = 8,  //  User specified an invalid transform size
	CUFFT_UNALIGNED_DATA = 9,  //  No longer used
	CUFFT_INCOMPLETE_PARAMETER_LIST = 10, //  Missing parameters in call
	CUFFT_INVALID_DEVICE = 11, //  Execution of a plan was on different GPU than plan creation
	CUFFT_PARSE_ERROR = 12, //  Internal plan database error 
	CUFFT_NO_WORKSPACE = 13  //  No workspace has been provided prior to plan execution
} cufftResult;


Read more at : http ://docs.nvidia.com/cuda/cufft/index.html#ixzz3e6fF4Fd3
Follow us : @GPUComputing on Twitter | NVIDIA on Facebook
*/
static void HandleError_cudFFT(cufftResult_t result, const char *file, int line)
{
	if (result != CUFFT_SUCCESS) 
	{ 
		printf("CU_FFT operation failed with result %d in file %s at line %d\n", result, file , line);
		exit(EXIT_FAILURE); 
	}
}

#define HANDLE_ERROR_KERNEL HandleError(cudaPeekAtLastError(),__FILE__, __LINE__ )

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL( a ) {if (a == NULL) { \
	printf("Host memory failed in %s at line %d\n", \
	__FILE__, __LINE__); \
	exit(EXIT_FAILURE); }}

#define HANDLE_CUFFT_ERROR ( result ) (HandleError_cudFFT(result, __FILE__, __LINE__ ))
#endif  // __BOOK_H__
