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


typedef float dataType;

using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "testing elementwise operations kernel in the GPU running..." << std::endl;

	int devCUDA = 0;
	std::uint64_t arrayLength = 100000;

	//=====================================================================

	HANDLE_ERROR(cudaSetDevice(devCUDA));
	/* initialize random seed: */
	srand(time(NULL));

	//generate arrays in CPU
	dataType* A = new dataType[arrayLength];
	dataType* B = new dataType[arrayLength];
	dataType* C = new dataType[arrayLength];


	for (uint64_t ii = 0; ii < arrayLength; ii++)
	{
		A[ii] = 1000 * (float(rand()) / (float)(RAND_MAX) - 0.5);
		B[ii] = 1000 * (float(rand()) / (float)(RAND_MAX)-0.5);
		C[ii] = 1000 * (float(rand()) / (float)(RAND_MAX)-0.5);
	}


	//generate arrays in GPU
	dataType *A_GPU, *B_GPU, *C_GPU;
	HANDLE_ERROR(cudaMalloc((void**)&(A_GPU), sizeof(dataType) * arrayLength));
	HANDLE_ERROR(cudaMalloc((void**)&(B_GPU), sizeof(dataType)* arrayLength));
	HANDLE_ERROR(cudaMalloc((void**)&(C_GPU), sizeof(dataType)* arrayLength));

	//copy data tp GPU
	HANDLE_ERROR(cudaMemcpy(A_GPU, A, arrayLength * sizeof(dataType), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(B_GPU, B, arrayLength * sizeof(dataType), cudaMemcpyHostToDevice));

	
	//test 1
	cout << "Test 1...";
	elementwiseOperationOutOfPlace(C_GPU, A_GPU, B_GPU, arrayLength, op_elementwise_type::minus);
	HANDLE_ERROR(cudaMemcpy(C, C_GPU, arrayLength * sizeof(dataType), cudaMemcpyDeviceToHost));
	//calculate in CPU
	dataType norm = 0;
	dataType aux = 0;
	for (uint64_t ii = 0; ii < arrayLength; ii++)
	{		
		aux = A[ii] - B[ii];
		aux = C[ii] - aux;
		norm = std::max(fabs(aux), norm);
	}

	if (norm > 1e-3)
	{
		cout << "ERROR: norm = " << norm << endl;
		return 2;
	}
	else{
		cout << "OK" << endl;
	}
	

	//test 2
	cout << "Test 2...";	
	elementwiseOperationInPlace(A_GPU, B_GPU, arrayLength, op_elementwise_type::divide);
	HANDLE_ERROR(cudaMemcpy(C, A_GPU, arrayLength * sizeof(dataType), cudaMemcpyDeviceToHost));
	//restore original A for future tests
	HANDLE_ERROR(cudaMemcpy(A_GPU, A, arrayLength * sizeof(dataType), cudaMemcpyHostToDevice));
	//calculate in CPU
	norm = 0;
	aux = 0;
	for (uint64_t ii = 0; ii < arrayLength; ii++)
	{
		aux = A[ii] / B[ii];
		aux = C[ii] - aux;
		norm = std::max(fabs(aux), norm);
	}

	if (norm > 1e-3)
	{
		cout << "ERROR: norm = " << norm << endl;
		return 2;
	}
	else{
		cout << "OK" << endl;
	}
	
	//test 3
	cout << "Test 3...";
	elementwiseOperationInPlace(A_GPU, B_GPU, arrayLength, op_elementwise_type::copy);
	HANDLE_ERROR(cudaMemcpy(C, A_GPU, arrayLength * sizeof(dataType), cudaMemcpyDeviceToHost));
	//restore original A for future tests
	HANDLE_ERROR(cudaMemcpy(A_GPU, A, arrayLength * sizeof(dataType), cudaMemcpyHostToDevice));
	//calculate in CPU
	norm = 0;
	aux = 0;
	for (uint64_t ii = 0; ii < arrayLength; ii++)
	{
		aux = B[ii];
		aux = C[ii] - aux;
		norm = std::max(fabs(aux), norm);
	}

	if (norm > 1e-3)
	{
		cout << "ERROR: norm = " << norm << endl;
		return 2;
	}
	else{
		cout << "OK" << endl;
	}



	//release memory
	HANDLE_ERROR(cudaFree(A_GPU));
	HANDLE_ERROR(cudaFree(B_GPU));
	HANDLE_ERROR(cudaFree(C_GPU));
	delete[] A;
	delete[] B;
	delete[] C;

}