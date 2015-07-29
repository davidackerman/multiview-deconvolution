/*
 * standardCUDAfunctions.h
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */

#ifndef STANDARDCUDAFUNCTIONS_H_
#define STANDARDCUDAFUNCTIONS_H_

//----------------------------------functions to decide whhich GPU to use-------------------------------


#ifdef _MSC_VER
	#define DLL_EXPORT  __declspec(dllexport)
#else
	#define DLL_EXPORT
#endif


extern "C" DLL_EXPORT int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
extern "C" DLL_EXPORT  int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
extern "C" DLL_EXPORT bool isDeviceCUDAusedByDisplay(int devCUDA);
extern "C" DLL_EXPORT  int getNumDevicesCUDA();
extern "C" DLL_EXPORT  void getNameDeviceCUDA(int devCUDA, char *name);
extern "C" DLL_EXPORT  long long int getMemDeviceCUDA(int devCUDA);
extern "C" DLL_EXPORT  void setDeviceCUDA(int devCUDA);

#endif /* STANDARDCUDAFUNCTIONS_H_ */
