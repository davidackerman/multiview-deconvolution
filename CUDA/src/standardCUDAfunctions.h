/*
 * standardCUDAfunctions.h
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */

#ifndef STANDARDCUDAFUNCTIONS_H_
#define STANDARDCUDAFUNCTIONS_H_

//----------------------------------functions to decide whhich GPU to use-------------------------------

extern "C" __declspec(dllexport) int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
extern "C" __declspec(dllexport) int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
extern "C" __declspec(dllexport) int getNumDevicesCUDA();
extern "C" __declspec(dllexport) void getNameDeviceCUDA(int devCUDA, char *name);
extern "C" __declspec(dllexport) long long int getMemDeviceCUDA(int devCUDA);

#endif /* STANDARDCUDAFUNCTIONS_H_ */
