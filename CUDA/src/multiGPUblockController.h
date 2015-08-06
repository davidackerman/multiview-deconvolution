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
#include <mutex>
#include <string>
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


class multiGPUblockController;//forward declaration for the typedef
//so I can call different deconvolution routines using the same multi-thread entry point
typedef  void (multiGPUblockController::*MV_deconv_fn)(size_t threadIdx);

//================================================================
class multiGPUblockController
{

public:

	//parameters for deconvolution
	paramDeconvolution paramDec;
	static const std::uint32_t goodFFTdims[65];

	//store original files in memory (in case we want to minimize I/O). This will contain the entire image
	multiviewImage<weightType> full_weights_mem;
	multiviewImage<psfType> full_psf_mem;
	multiviewImage<outputType> full_img_mem;

    //constructor/destructor
    multiGPUblockController();
	multiGPUblockController(std::string filenameXML);
	~multiGPUblockController();

    //short set/get methods
	std::uint64_t numElements();
	int memoryRequirements(){ return 4 * paramDec.Nviews + 5 + 4; }//memoryRequirements * imgSize * sizeof(float32) is the amount of memory required in a GPU (empirical formula). The problem is that cuFFT needs more workspace depending on the factors of the image size (so we need to be on the save side)
	int getImageDimensions();
	size_t getNumGPU() const{ return GPUinfoVec.size(); };
	int writeDeconvoutionResult(const std::string& filename);
	int writeDeconvoutionResultRaw(const std::string& filename);
	int writeDeconvoutionResult_uint16(const std::string& filename);
	int getDimBlockPartition(){ return dimBlockParition; };

    //methods
	void queryGPUs(int maxNumber = -1);
	int findBestBlockPartitionDimension();
	void findMaxBlockPartitionDimensionPerGPU();
	int findBestBlockPartitionDimension_inMem();
	void findMaxBlockPartitionDimensionPerGPU_inMem();

	int runMultiviewDeconvoution(MV_deconv_fn p);//main function to start distirbuting multiview deconvolution to different blocks		
	void copyBlockResultToJ(const imgTypeDeconvolution* Jsrc, const uint32_t blockDims[MAX_DATA_DIMS], int64_t Joffset, int64_t Boffset, int64_t numPlanes);
	static std::uint32_t ceilToGoodFFTsize(std::uint32_t n);
	static std::uint32_t padToGoodFFTsize(std::uint32_t n);
	void calculateWeights();//main function to calculate weights on each image using multiple GPU
	
	//functions to perform different kinds of deconvolution on each block	
	void multiviewDeconvolutionBlockWise_fromFile(size_t threadIdx);//main function for each GPU to process different blocks
	void multiviewDeconvolutionBlockWise_fromMem(size_t threadIdx);

    //debug functions
	void debug_listGPUs();
	void debug_setGPUmaxSizeDimBlockPartition(size_t pos, std::uint32_t ss){ 
		std::cout << "==============DEBUGGING: manually modiying findMaxBlockPartitionDimensionPerGPU value to test with two GPUs==================" << std::endl;
        GPUinfoVec[pos].maxSizeDimBlockPartition = ss; 
        };

protected:	
    //stores all the necessary information to partition in blocks for each computational unit
	std::vector<GPUinfo> GPUinfoVec;    
	

	//to store final output
	imgTypeDeconvolution *J;

    //which dimension we use to generate blocks (the one where the PSF is smaller so we minimize padding)
	int dimBlockParition;
	std::uint32_t padBlockPartition;//largest size of a psf in the dimBlockPartitionDimension
	std::uint32_t imgDims[MAX_DATA_DIMS];


	std::mutex              g_lock_offset;//to keep block offset for each thread
	int64_t offsetBlockPartition;

    //methods
	void findMaxBlockPartitionDimensionPerGPU(size_t pos);   	
	void findMaxBlockPartitionDimensionPerGPU_inMem(size_t pos);

	void calculateWeightsSingleView(size_t threadIdx);
private:

	//calculate DCT weioghts all at once
	void calculateWeightsSingleView_allAtOnce(int view, float anisotropyZ);
	//calculate DCT weights into blocks
	void calculateWeightsSingleView_lowMem(int view, float anisotropyZ, int64_t availMem);

};


#endif 
