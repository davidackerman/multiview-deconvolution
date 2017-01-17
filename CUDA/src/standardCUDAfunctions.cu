/*
 * standardCUDAfunctions.cu
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */
#include "book.h"
#include "cuda.h"
#include "standardCUDAfunctions.h"

//==============================================
int getCUDAcomputeCapabilityMajorVersion(int devCUDA)
{
	int major = 0, minor = 0;
	cuDeviceComputeCapability(&major, &minor,devCUDA);

	return major;
}
int getCUDAcomputeCapabilityMinorVersion(int devCUDA)
{
	int major = 0, minor = 0;
	cuDeviceComputeCapability(&major, &minor,devCUDA);

	return minor;
}

int getNumDevicesCUDA()
{
	int count = 0;
	HANDLE_ERROR(cudaGetDeviceCount ( &count ));
	return count;
}
void getNameDeviceCUDA(int devCUDA, char* name)
{
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));

	memcpy(name,prop.name,sizeof(char)*256);
}

#include <iostream>

bool isDeviceCUDAusedByDisplay(int devCUDA)
{
	int has_timeout;
	HANDLE_ERROR( cudaDeviceGetAttribute(&has_timeout, cudaDevAttrKernelExecTimeout, devCUDA) );
	//std::cout << has_timeout << std::endl;
	return (has_timeout > 0);
}

long long int getMemDeviceCUDA(int devCUDA)
{
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));
	return ((long long int)prop.totalGlobalMem);
}

long long int getAvailableMemDeviceCUDA(int devCUDA)
{
	setDeviceCUDA(devCUDA);
	size_t free, total;
	HANDLE_ERROR(cudaMemGetInfo(&free, &total));
	return ((long long int)free);
}

void setDeviceCUDA(int devCUDA)
{
	HANDLE_ERROR(cudaSetDevice(devCUDA));	
}

void resetDeviceCUDA(int devCUDA)
{
	HANDLE_ERROR(cudaSetDevice(devCUDA));
	HANDLE_ERROR(cudaDeviceReset());
}