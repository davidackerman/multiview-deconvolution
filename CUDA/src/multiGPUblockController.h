/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  multiGPUblockCOntroller.h
*
*  Created on : June 11th, 2015
* Author : Fernando Amat
*
* \brief main interface to control splitting blocks to different resources and stitching them back together 
*/

#ifndef __MULTI_GPU_BLOCK_CONTROLLER_HEADER_H__
#define __MULTI_GPU_BLOCK_CONTROLLER_HEADER_H__

#include <cstdint>
#include <vector>
#include <atomic>
#include "paramDeconvolution.h"
#include "multiviewDeconvolution.h"





struct GPUinfo
{
	std::int64_t mem;//in bytes
	int devCUDA;
	std::string devName;
	std::uint32_t maxSizeDimBlockPartition;//maximum number of elements along the direction of the dimBlockParition

	GPUinfo()
	{
		maxSizeDimBlockPartition = 0;
	}
};

//================================================================
class multiGPUblockController
{

public:

	//parameters for deconvolution
	paramDeconvolution paramDec;

    //constructor/destructor
    multiGPUblockController();
	~multiGPUblockController();

    //short set/get methods
	std::uint64_t numElements();
	int memoryRequirements(){ return 4 * paramDec.Nviews + 4; }//memoryRequirements * imgSize * sizeof(float32) is the amount of memory required in a GPU (empirical formula)

    //methods
	void queryGPUs(); 
	int findBestBlockPartitionDimension();
	void findMaxBlockPartitionDimensionPerGPU();
	int runMultiviewDeconvoution();//main function to start distirbuting multiview deconvolution to different blocks
	int getImageDimensions();


    //debug functions
	void debug_listGPUs();

protected:	
    //stores all the necessary information to partition in blocks for each computational unit
	std::vector<GPUinfo> GPUinfoVec;    

	imgTypeDeconvolution *J;//to store final output

    //which dimension we use to generate blocks (the one where the PSF is smaller so we minimize padding)
	int dimBlockParition;
	std::uint32_t padBlockPartition;//largest size of a psf in the dimBlockPartitionDimension
	std::uint32_t imgDims[MAX_DATA_DIMS];

    //methods
	void findMaxBlockPartitionDimensionPerGPU(size_t pos);   
	void multiviewDeconvolutionBlockWise(size_t threadIdx, std::atomic<int64_t> *offsetBlockPartition);//main function for each GPU to process different blocks
private:

};

#endif 