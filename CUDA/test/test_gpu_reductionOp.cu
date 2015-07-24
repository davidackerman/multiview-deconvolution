/*
*
* Authors: Fernando Amat
*  test_gpu_elementwiseOp.cpp
*
*  Created on : June 5th, 2015
* Author : Fernando Amat
*
* \brief testing GPU kernels to perform pointwise operations
*
*/

#include <cstdint>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <algorithm>
#include "commonCUDA.h"
#include "cuda.h"
#include "book.h"


typedef double dataType;

using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "testing reduction operations kernel in the GPU running..." << std::endl;

	int devCUDA = 0;
	std::uint64_t arrayLength = 1000000;

	//=====================================================================

	HANDLE_ERROR(cudaSetDevice(devCUDA));
	/* initialize random seed: */
	srand(time(NULL));

	//generate arrays in CPU
	dataType* A = new dataType[arrayLength];	

	for (uint64_t ii = 0; ii < arrayLength; ii++)
	{
		A[ii] = 10 * (float(rand()) / (float)(RAND_MAX) - 0.5);		
	}


	//generate arrays in GPU
	dataType *A_GPU;
	HANDLE_ERROR(cudaMalloc((void**)&(A_GPU), sizeof(dataType) * arrayLength));
	

	//copy data tp GPU
	HANDLE_ERROR(cudaMemcpy(A_GPU, A, arrayLength * sizeof(dataType), cudaMemcpyHostToDevice));
	

	
	//test 1
	cout << "Test 1: add vector values...";
	dataType result_GPU = reductionOperation(A_GPU, arrayLength, op_reduction_type::add);
	
	//calculate in CPU
	dataType result_CPU = 0;
	for (uint64_t ii = 0; ii < arrayLength; ii++)
	{		
		result_CPU += A[ii];
	}

	if (fabs(result_CPU-result_GPU) > 1e-3)
	{
		cout << "ERROR: norm = " << fabs(result_CPU - result_GPU) << endl;
		return 2;
	}
	else{
		cout << "OK" << endl;
	}
	
	//test 2
	cout << "Test 1: max vector values...";
	result_GPU = reductionOperation(A_GPU, arrayLength, op_reduction_type::max_elem);

	//calculate in CPU
	result_CPU = numeric_limits<dataType>::min();
	for (uint64_t ii = 0; ii < arrayLength; ii++)
	{
		result_CPU = max(A[ii], result_CPU);
	}

	if (fabs(result_CPU - result_GPU) > 1e-3)
	{
		cout << "ERROR: norm = " << fabs(result_CPU - result_GPU) << endl;
		return 2;
	}
	else{
		cout << "OK" << endl;
	}
	
	//test 3
	cout << "Test 1: min vector values...";
	result_GPU = reductionOperation(A_GPU, arrayLength, op_reduction_type::min_elem);

	//calculate in CPU
	result_CPU = numeric_limits<dataType>::max();
	for (uint64_t ii = 0; ii < arrayLength; ii++)
	{
		result_CPU = min(A[ii], result_CPU);
	}

	if (fabs(result_CPU - result_GPU) > 1e-3)
	{
		cout << "ERROR: norm = " << fabs(result_CPU - result_GPU) << endl;
		return 2;
	}
	else{
		cout << "OK" << endl;
	}



	//release memory
	HANDLE_ERROR(cudaFree(A_GPU));	
	delete[] A;
	
	

}