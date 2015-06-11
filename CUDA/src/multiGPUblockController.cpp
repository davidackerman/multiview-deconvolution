/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  multiGPUblockCOntroller.cpp
*
*  Created on : June 11th, 2015
* Author : Fernando Amat
*
* \brief main interface to control splitting blocks to different resources and stitching them back together
*/

#include "multiGPUblockController.h"
#include "standardCUDAfunctions.h"
#include "multiviewDeconvolution.h"




using namespace std;

multiGPUblockController::multiGPUblockController()
{
	dimBlockParition = -1;
}
//============================================================
multiGPUblockController::~multiGPUblockController()
{

}
//============================================================
void multiGPUblockController::queryGPUs()
{
	int nGPU = getNumDevicesCUDA();

	char buffer[1024];
	for (int ii = 0; ii < nGPU; ii++)
	{
		if (getCUDAcomputeCapabilityMajorVersion(ii) < 2.0)
			continue;

		GPUinfoVec.push_back(GPUinfo());
		GPUinfoVec.back().devCUDA = ii;
		GPUinfoVec.back().mem = getMemDeviceCUDA(ii);
		getNameDeviceCUDA(ii, buffer);
		GPUinfoVec.back().devName = string(buffer);
	}
}

//========================================================
int multiGPUblockController::findBestBlockPartitionDimension()
{
	string filename;
	uint32_t maxDims[MAX_DATA_DIMS], auxDims[MAX_DATA_DIMS];
	memset(maxDims, 0, sizeof(uint32_t)* MAX_DATA_DIMS);
	for (int ii = 0; ii < paramDec.Nviews; ii++)
	{
        //load psf header to find dimensions
		filename = multiviewImage<imgTypeDeconvolution>::recoverFilenamePatternFromString(paramDec.filePatternPSF, ii + 1);
		int err = multiviewImage<imgTypeDeconvolution>::getImageDimensionsFromHeader(filename, auxDims);
		if (err > 0)
			return err;

		for (int jj = 0; jj < MAX_DATA_DIMS; jj++)
		{
			if (maxDims[jj] < auxDims[jj])
			{
				maxDims[jj] = auxDims[jj];
			}
		}
	}

	dimBlockParition = 0;
	padBlockPartition = 4294967295;

	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{
		if (padBlockPartition > maxDims[ii])
		{
			padBlockPartition = maxDims[ii];
			dimBlockParition = ii;
		}
	}

	return 0;
}
//================================================================
void multiGPUblockController::findMaxBlockPartitionDimensionPerGPU()
{
	for (size_t ii = 0; ii < GPUinfoVec.size(); ii++)
		findMaxBlockPartitionDimensionPerGPU(ii);
}

//================================================================
void multiGPUblockController::findMaxBlockPartitionDimensionPerGPU(size_t pos)
{
	uint32_t auxDims[MAX_DATA_DIMS];

	string filename = multiviewImage<imgTypeDeconvolution>::recoverFilenamePatternFromString(paramDec.filePatternImg, 1);
	int err = multiviewImage<imgTypeDeconvolution>::getImageDimensionsFromHeader(filename, auxDims);

	int64_t sliceSize = sizeof(imgTypeDeconvolution);
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{   
		if (ii != dimBlockParition)
			sliceSize *= (uint64_t)(auxDims[ii]);
	}

	GPUinfoVec[pos].maxSizeDimBlockPartition = GPUinfoVec[pos].mem / (sliceSize * memoryRequirements());
}
