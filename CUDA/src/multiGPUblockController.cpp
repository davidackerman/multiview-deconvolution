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

#include <thread>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <assert.h>
#include "multiGPUblockController.h"
#include "external/xmlParser/xmlParser.h"
#include "external/xmlParser/svlStrUtils.h"
#include "standardCUDAfunctions.h"
#include "klb_ROI.h"
#include "klb_imageIO.h"
#include "klb_Cwrapper.h"
#include "weigthsBlurryMeasure.h"
#include "commonCUDA.h"



//uncomment this to time elements
//#define PROFILE_CODE_LR_GPU
	

typedef std::chrono::high_resolution_clock Clock;
using namespace std;

//these numbers only have 2 or 3 as factors
///this number has to be smaller than 65/
#define NUM_POSSIBLE_SIZES (65)
const std::uint32_t multiGPUblockController::goodFFTdims[65] = { 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243, 256, 288, 324, 384, 432, 486, 512, 576, 648, 729, 768, 864, 972, 1024, 1152, 1296, 1458, 1536, 1728, 1944, 2048, 2187, 2304, 2592, 2916, 3072, 3456, 3888, 4096, 4374, 4608, 5184, 5832, 6144, 6561, 6912, 7776, 8192};


//const std::uint32_t multiGPUblockController::goodFFTdims[65] = {1,2,3,4,5,8,9,16,25,27,32,64,81,125,128,243,256,512,625,729,1024,2048,2187,3125,4096,6561,8192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

multiGPUblockController::multiGPUblockController()
{
	dimBlockParition = -1;
	J = NULL;
}

multiGPUblockController::multiGPUblockController(string filenameXML)
{
	J = NULL;
	dimBlockParition = -1;

	//set default parameters
	paramDec.setDefaultParam();

	//check if file exists
	ifstream fin(filenameXML.c_str());
	if (!fin.is_open())
	{
		cout << "ERROR: multiGPUblockController: file " << filenameXML << " cannot be opened" << endl;
	}
	fin.close();

	//parse file
	XMLNode xMainNode = XMLNode::openFileHelper(filenameXML.c_str(), "document");
	
	paramDec.Nviews = xMainNode.nChildNode("view");
	int position;

	for (int ii = 0; ii < paramDec.Nviews; ii++)
	{
		position = ii;
		XMLNode node = xMainNode.getChildNode("view", &position);

		XMLCSTR aux = node.getAttribute("imgFilename");		
		assert(aux != NULL);		
		paramDec.fileImg.push_back(string(aux));


		aux = node.getAttribute("psfFilename");
		assert(aux != NULL);
		paramDec.filePSF.push_back(string(aux));


		vector<float> vv;
		aux = node.getAttribute("A");
		assert(aux != NULL);
		parseString<float>(string(aux), vv);
		paramDec.Acell.push_back(vv);
		vv.clear();
		
		aux = node.getAttribute("verbose");
		vector<int> ll;
		if (aux != NULL)
		{
			parseString<int>(string(aux), ll);
			paramDec.verbose = ll[0]; 
			ll.clear();
		}

	}

	//read deconvolution parameters
	position = 0;
	XMLNode node = xMainNode.getChildNode("deconvolution", &position);

	XMLCSTR aux = node.getAttribute("numIter");
	vector<int> ll;
	if (aux != NULL)
	{
		parseString<int>(string(aux), ll);
		paramDec.numIters = ll[0];
		ll.clear();
	}

	aux = node.getAttribute("imBackground");
	vector<float> vv;
	if (aux != NULL)
	{
		parseString<float>(string(aux), vv);
		paramDec.imgBackground = vv[0];
		vv.clear();
	}

	aux = node.getAttribute("lambdaTV");
	if (aux != NULL)
	{
		parseString<float>(string(aux), vv);
		paramDec.lambdaTV = vv[0];
		vv.clear();
	}

	paramDec.anisotropyZ = paramDec.getAnisotropyZfromAffine();
}

//============================================================
multiGPUblockController::~multiGPUblockController()
{
	if (J != NULL)
		delete[] J;
}
//============================================================
void multiGPUblockController::queryGPUs(int maxNumber)
{
	int nGPU = getNumDevicesCUDA();
	if (maxNumber <= 0)
		maxNumber = nGPU;//all available ones

	char buffer[1024];
	for (int ii = 0; ii < min(nGPU, maxNumber); ii++)
	{
		if (getCUDAcomputeCapabilityMajorVersion(ii) < 2.0)
			continue;
		if (isDeviceCUDAusedByDisplay(ii))		
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
//========================================================
int multiGPUblockController::findBestBlockPartitionDimension_inMem()
{
	string filename;
	uint32_t maxDims[MAX_DATA_DIMS];
	memset(maxDims, 0, sizeof(uint32_t)* MAX_DATA_DIMS);
	for (int ii = 0; ii < paramDec.Nviews; ii++)
	{		
		for (int jj = 0; jj < MAX_DATA_DIMS; jj++)
		{
			if (maxDims[jj] < full_psf_mem.dimsImgVec[ii].dims[jj])
			{
				maxDims[jj] = full_psf_mem.dimsImgVec[ii].dims[jj];
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
void multiGPUblockController::findMaxBlockPartitionDimensionPerGPU_inMem()
{
	for (size_t ii = 0; ii < GPUinfoVec.size(); ii++)
		findMaxBlockPartitionDimensionPerGPU_inMem(ii);
}

//================================================================
void multiGPUblockController::findMaxBlockPartitionDimensionPerGPU(size_t pos)
{
	
	int err = getImageDimensions();
	if (err > 0)
		exit(3);

	int64_t sliceSize = sizeof(imgTypeDeconvolution);
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{   
		if (ii != dimBlockParition)
			sliceSize *= (uint64_t)(imgDims[ii]);
	}

	GPUinfoVec[pos].maxSizeDimBlockPartition = GPUinfoVec[pos].mem / (sliceSize * memoryRequirements());

    //find a good value for FFT
	GPUinfoVec[pos].maxSizeDimBlockPartition = ceilToGoodFFTsize(GPUinfoVec[pos].maxSizeDimBlockPartition);
}
//================================================================
void multiGPUblockController::findMaxBlockPartitionDimensionPerGPU_inMem(size_t pos)
{

	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{
		imgDims[ii] = full_img_mem.dimsImgVec[pos].dims[ii];
	}


	int64_t sliceSize = sizeof(imgTypeDeconvolution);
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{
		if (ii != dimBlockParition)
			sliceSize *= (uint64_t)(imgDims[ii]);
	}

	GPUinfoVec[pos].maxSizeDimBlockPartition = GPUinfoVec[pos].mem / (sliceSize * memoryRequirements());

	//find a good value for FFT
	GPUinfoVec[pos].maxSizeDimBlockPartition = ceilToGoodFFTsize(GPUinfoVec[pos].maxSizeDimBlockPartition);

}

//===================================
uint32_t multiGPUblockController::ceilToGoodFFTsize(uint32_t n)
{
	for (int ii = 1; ii < NUM_POSSIBLE_SIZES; ii++)
	{
		if (goodFFTdims[ii] > n)
			return goodFFTdims[ii - 1];
	}

	return n;//number is too high
}

//===================================
uint32_t multiGPUblockController::padToGoodFFTsize(uint32_t n)
{
	//find the first set of dimensions that fit
	for (int ii = 1; ii < NUM_POSSIBLE_SIZES; ii++)
	{
		if (goodFFTdims[ii] >= n)
			return goodFFTdims[ii];
	}

	return n;//number is too high
}
//================================================================
int multiGPUblockController::runMultiviewDeconvoution(MV_deconv_fn p)
{
    //calculate image size
	if (full_img_mem.getNumberOfViews() > 0)
	{

		for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		{
			imgDims[ii] = full_img_mem.dimsImgVec[0].dims[ii];
		}
	}
	else{

		int err = getImageDimensions();
		if (err > 0)
			return err;
	}
	uint64_t nImg = numElements();
	

	//allocate memory for output
	J = new imgTypeDeconvolution[nImg];

    //define atomic offset variable to keep count
	offsetBlockPartition  = 0;//defines the beginning for each block
	

    //launch threads (each thread will write into J disjointly)
	// start the working threads
	std::vector<std::thread> threads;	
	for (size_t ii = 0; ii < GPUinfoVec.size(); ++ii)
	{		
		threads.push_back(std::thread(p, this, ii));
	}

	//wait for the workers to finish
	for (auto& t : threads)
		t.join();

	return 0;
}

//================================================================
void multiGPUblockController::calculateWeights()
{
	//resize vector of weights
	full_weights_mem.resize(paramDec.Nviews);
	//define atomic variable to keep count of which view to process
	offsetBlockPartition = 0;

	//launch threads (each thread will write into J disjointly)
	// start the working threads
	std::vector<std::thread> threads;
	for (size_t ii = 0; ii < GPUinfoVec.size(); ++ii)
	{
		threads.push_back(std::thread(&multiGPUblockController::calculateWeightsSingleView, this, ii));
	}

	//wait for the workers to finish
	for (auto& t : threads)
		t.join();
}

//==========================================================
void multiGPUblockController::multiviewDeconvolutionBlockWise_fromFile(size_t threadIdx)
{
	const int64_t PSFpadding = 1+ padBlockPartition / 2;//quantity that we have to pad on each side to avoid edge effects
	const int64_t chunkSize = std::min((int64_t)(GPUinfoVec[threadIdx].maxSizeDimBlockPartition) - 2 * PSFpadding, int64_t(imgDims[dimBlockParition]));//effective size where we are calculating LR	
	const int devCUDA = GPUinfoVec[threadIdx].devCUDA;
    int64_t JoffsetIni, JoffsetEnd;//useful slices in J (global final outout) are from [JoffsetIni,JoffsetEnd)
	int64_t BoffsetIni;//useful slices in ROI (local input) are from [BoffsetIni, BoffsetEnd]. Thus, at the end we copy J(:,.., JoffsetIni:JoffsetEnd-1,:,..:) = Jobj->J(:,.., BoffsetIni:BoffsetEnd-1,:,..:)
	uint32_t xyzct[KLB_DATA_DIMS];
    

    //copy value to define klb ROI
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		xyzct[ii] = imgDims[ii];
	for (int ii = MAX_DATA_DIMS; ii < KLB_DATA_DIMS; ii++)
		xyzct[ii] = 1;

    //make sure we have useful are of computation
	if (chunkSize <= 0)
	{
		cout << "WARNING: device CUDA " << GPUinfoVec[threadIdx].devCUDA << " with GPU " << GPUinfoVec[threadIdx].devName<<" cannot process enough effective planes after padding" << endl;
		return;
	}

    //set cuda device for this thread
	setDeviceCUDA(devCUDA);

#ifdef _DEBUG
	cout << "Thread " << threadIdx << " initializing multiviewDeconvolution object" << endl;
#endif
    //instatiate multiview deconvolution object
	multiviewDeconvolution<imgTypeDeconvolution> *Jobj;
	Jobj = new multiviewDeconvolution<imgTypeDeconvolution>;
	//set number of views
	Jobj->setNumberOfViews(paramDec.Nviews);

	//read PSF
#ifdef _DEBUG
	cout << "Thread " << threadIdx << " reading PSF" << endl;
#endif
	string filename;
	int err;
	for (int ii = 0; ii < paramDec.Nviews; ii++)
	{
		filename = multiviewImage<float>::recoverFilenamePatternFromString(paramDec.filePatternPSF, ii + 1);
		err = Jobj->readImage(filename, ii, std::string("psf"));//this function should just read image
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			exit(err);
		}		
	}

    //define ROI dimensions	
#ifdef _DEBUG
	cout << "Thread " << threadIdx << " allocating initial workspace" << endl;
#endif
	uint32_t blockDims[MAX_DATA_DIMS];//maximum size of a block to preallocate memory
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		blockDims[ii] = xyzct[ii];//image size except a long the dimension of partition
	blockDims[dimBlockParition] = std::min(GPUinfoVec[threadIdx].maxSizeDimBlockPartition, imgDims[dimBlockParition]);;

	const bool useWeights = (paramDec.filePatternWeights.length() > 1 );    
	Jobj->allocate_workspace_init_multiGPU(blockDims, useWeights);

    //main loop
	while (1)
	{
		std::unique_lock<std::mutex> locker(g_lock_offset);//obtain lock to calculate block

		JoffsetIni = offsetBlockPartition;

		//check if we have more slices
		if (JoffsetIni >= imgDims[dimBlockParition])
			break;

        //generate ROI to define block        
		klb_ROI ROI;
		ROI.defineSlice(JoffsetIni, dimBlockParition, xyzct);//this would be a slice through dimsBlockParitiondimension equal to offset
		BoffsetIni = PSFpadding;
		if (JoffsetIni >= PSFpadding)//we cannot gain anything
		{
			ROI.xyzctLB[dimBlockParition] -= PSFpadding;
		}
		else{//we can process more slices in the block since we are at the beggining
			ROI.xyzctLB[dimBlockParition] = 0;
			BoffsetIni -= (PSFpadding - JoffsetIni);
		}

		JoffsetEnd = JoffsetIni + chunkSize + PSFpadding - BoffsetIni; 
		JoffsetEnd = std::min((uint32_t)JoffsetEnd, xyzct[dimBlockParition]);//make sure we do not go over the end of the image

		offsetBlockPartition = JoffsetEnd;//update offset counter

#ifdef _DEBUG
		cout << "Thread " << threadIdx << " processing block with offset ini " << JoffsetIni << " to " <<JoffsetEnd<< endl;
#endif
	    locker.unlock();//release lock

		ROI.xyzctUB[dimBlockParition] = JoffsetEnd + PSFpadding - 1;
		ROI.xyzctUB[dimBlockParition] = std::min(ROI.xyzctUB[dimBlockParition], xyzct[dimBlockParition] - 1);//make sure we do not go over the end of the image		

        //read image and weights ROI
		for (int ii = 0; ii < paramDec.Nviews; ii++)
		{
			filename = multiviewImage<float>::recoverFilenamePatternFromString(paramDec.filePatternImg, ii + 1);		
            err = Jobj->readROI(filename, ii, std::string("img"), ROI);//this function should just read image
			if (err > 0)
			{
				cout << "ERROR: reading file " << filename << endl;
				exit(err);
			}
			filename = multiviewImage<float>::recoverFilenamePatternFromString(paramDec.filePatternWeights, ii + 1);
			err = Jobj->readROI(filename, ii, std::string("weight"), ROI);//this function should just read image
			if (err > 0)
			{
				cout << "ERROR: reading file " << filename << endl;
				exit(err);
			}

			//the last block has to be padded at the end to match the block dimensions
			if (ROI.getSizePixels(dimBlockParition) != blockDims[dimBlockParition])
			{
				Jobj->padArrayWithZeros(blockDims, ii, "weight");
				Jobj->padArrayWithZeros(blockDims, ii, "img");
			}
		}

        

        //update workspace with ROI image and weights	
#ifdef _DEBUG
		cout << "Thread " << threadIdx << " updating workspace for deconvolution" << endl;
#endif	
		Jobj->allocate_workspace_update_multiGPU(paramDec.imgBackground, useWeights);

        //calculate multiview deconvolution
#ifdef _DEBUG
		cout << "Thread " << threadIdx << " running deconvolution" << endl;
#endif

#ifdef PROFILE_CODE_LR_GPU
		auto t1 = Clock::now();
#endif
		Jobj->deconvolution_LR_TV(paramDec.numIters, paramDec.lambdaTV);

#ifdef PROFILE_CODE_LR_GPU
		//deconvolution_LR_TV calls cudaFree at the end which is synchronous so timing is accurate
		auto t2 = Clock::now();
		cout << "Thread " << threadIdx << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << " for one block with " << paramDec.numIters << " iters" << endl;
#endif

        //copy block result to J
		Jobj->copyDeconvoutionResultToCPU();
		copyBlockResultToJ(Jobj->getJpointer(), blockDims, JoffsetIni, BoffsetIni, JoffsetEnd - JoffsetIni);
	}


	delete Jobj;
}


//==========================================================
void multiGPUblockController::multiviewDeconvolutionBlockWise_fromMem(size_t threadIdx)
{
	const int64_t PSFpadding = 1 + padBlockPartition / 2;//quantity that we have to pad on each side to avoid edge effects
	const int64_t chunkSize = std::min((int64_t)(GPUinfoVec[threadIdx].maxSizeDimBlockPartition) - 2 * PSFpadding, int64_t(imgDims[dimBlockParition]));//effective size where we are calculating LR	
	const int devCUDA = GPUinfoVec[threadIdx].devCUDA;
	int64_t JoffsetIni, JoffsetEnd;//useful slices in J (global final outout) are from [JoffsetIni,JoffsetEnd)
	int64_t BoffsetIni;//useful slices in ROI (local input) are from [BoffsetIni, BoffsetEnd]. Thus, at the end we copy J(:,.., JoffsetIni:JoffsetEnd-1,:,..:) = Jobj->J(:,.., BoffsetIni:BoffsetEnd-1,:,..:)
	uint32_t xyzct[KLB_DATA_DIMS];


	//copy value to define klb ROI
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		xyzct[ii] = imgDims[ii];
	for (int ii = MAX_DATA_DIMS; ii < KLB_DATA_DIMS; ii++)
		xyzct[ii] = 1;

	//make sure we have useful are of computation
	if (chunkSize <= 0)
	{
		cout << "WARNING: device CUDA " << GPUinfoVec[threadIdx].devCUDA << " with GPU " << GPUinfoVec[threadIdx].devName << " cannot process enough effective planes after padding" << endl;
		return;
	}

	//set cuda device for this thread
	setDeviceCUDA(devCUDA);

#ifdef _DEBUG
	cout << "Thread " << threadIdx << " initializing multiviewDeconvolution object" << endl;
#endif
	//instatiate multiview deconvolution object
	multiviewDeconvolution<imgTypeDeconvolution> *Jobj;
	Jobj = new multiviewDeconvolution<imgTypeDeconvolution>;
	//set number of views
	Jobj->setNumberOfViews(paramDec.Nviews);

	//set the pointer to PSF
#ifdef _DEBUG
	cout << "Thread " << threadIdx << " reading PSF" << endl;
#endif	
	int err;
	Jobj->psf.resize(paramDec.Nviews);
	for (int ii = 0; ii < paramDec.Nviews; ii++)
	{				
		Jobj->psf.copyView_extPtr_to_CPU(ii, full_psf_mem.getPointer_CPU(ii), full_psf_mem.dimsImgVec[ii].dims);
	}

	//define ROI dimensions	
#ifdef _DEBUG
	cout << "Thread " << threadIdx << " allocating initial workspace" << endl;
#endif
	uint32_t blockDims[MAX_DATA_DIMS];//maximum size of a block to preallocate memory
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		blockDims[ii] = xyzct[ii];//image size except a long the dimension of partition
	blockDims[dimBlockParition] = std::min(GPUinfoVec[threadIdx].maxSizeDimBlockPartition, imgDims[dimBlockParition]);;

	const bool useWeights = (full_weights_mem.getNumberOfViews() > 1);
	Jobj->allocate_workspace_init_multiGPU(blockDims, useWeights);

	//main loop
	while (1)
	{
		std::unique_lock<std::mutex> locker(g_lock_offset);//obtain lock to calculate block

		JoffsetIni = offsetBlockPartition;

		//check if we have more slices
		if (JoffsetIni >= imgDims[dimBlockParition])
			break;

		//generate ROI to define block        
		klb_ROI ROI;
		ROI.defineSlice(JoffsetIni, dimBlockParition, xyzct);//this would be a slice through dimsBlockParitiondimension equal to offset
		BoffsetIni = PSFpadding;
		if (JoffsetIni >= PSFpadding)//we cannot gain anything
		{
			ROI.xyzctLB[dimBlockParition] -= PSFpadding;
		}
		else{//we can process more slices in the block since we are at the beggining
			ROI.xyzctLB[dimBlockParition] = 0;
			BoffsetIni -= (PSFpadding - JoffsetIni);
		}

		JoffsetEnd = JoffsetIni + chunkSize + PSFpadding - BoffsetIni;
		JoffsetEnd = std::min((uint32_t)JoffsetEnd, xyzct[dimBlockParition]);//make sure we do not go over the end of the image

		offsetBlockPartition = JoffsetEnd;//update offset counter

#ifdef _DEBUG
		cout << "Thread " << threadIdx << " processing block with offset ini " << JoffsetIni << " to " << JoffsetEnd << endl;
#endif
		locker.unlock();//release lock

		ROI.xyzctUB[dimBlockParition] = JoffsetEnd + PSFpadding - 1;
		ROI.xyzctUB[dimBlockParition] = std::min(ROI.xyzctUB[dimBlockParition], xyzct[dimBlockParition] - 1);//make sure we do not go over the end of the image		

		//copy image and weights ROI
		for (int ii = 0; ii < paramDec.Nviews; ii++)
		{			
			Jobj->img.copyROI(full_img_mem.getPointer_CPU(ii), full_img_mem.dimsImgVec[ii].dims, full_img_mem.dimsImgVec[ii].ndims, ii, ROI);//this function should just copy image			
			
			Jobj->weights.copyROI(full_weights_mem.getPointer_CPU(ii), full_weights_mem.dimsImgVec[ii].dims, full_weights_mem.dimsImgVec[ii].ndims, ii, ROI);//this function should just copy image
			
			//the last block has to be padded at the end to match the block dimensions
			if (ROI.getSizePixels(dimBlockParition) != blockDims[dimBlockParition])
			{
				Jobj->padArrayWithZeros(blockDims, ii, "weight");
				Jobj->padArrayWithZeros(blockDims, ii, "img");
			}
		}



		//update workspace with ROI image and weights	
#ifdef _DEBUG
		cout << "Thread " << threadIdx << " updating workspace for deconvolution" << endl;
#endif	
		Jobj->allocate_workspace_update_multiGPU(paramDec.imgBackground, useWeights);

		//calculate multiview deconvolution
#ifdef _DEBUG
		cout << "Thread " << threadIdx << " running deconvolution" << endl;
#endif

#ifdef PROFILE_CODE_LR_GPU
		auto t1 = Clock::now();
#endif
		Jobj->deconvolution_LR_TV(paramDec.numIters, paramDec.lambdaTV);

#ifdef PROFILE_CODE_LR_GPU
		//deconvolution_LR_TV calls cudaFree at the end which is synchronous so timing is accurate
		auto t2 = Clock::now();
		cout << "Thread " << threadIdx << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << " for one block with " << paramDec.numIters << " iters" << endl;
#endif

		//copy block result to J
		Jobj->copyDeconvoutionResultToCPU();
		copyBlockResultToJ(Jobj->getJpointer(), blockDims, JoffsetIni, BoffsetIni, JoffsetEnd - JoffsetIni);
	}


	delete Jobj;
}

//==========================================================
void multiGPUblockController::calculateWeightsSingleView(size_t threadIdx)
{
	
	const int devCUDA = GPUinfoVec[threadIdx].devCUDA;
	

	//set cuda device for this thread
	setDeviceCUDA(devCUDA);

	int view;
	//main loop
	while (1)
	{
		std::unique_lock<std::mutex> locker(g_lock_offset);//obtain lock to calculate block
		view = offsetBlockPartition;
		offsetBlockPartition++;
		locker.unlock();//release lock

		if (view >= paramDec.Nviews)
			break;

		//set dimensions for weights
		full_weights_mem.setImgDims(view, full_img_mem.dimsImgVec[view]);


		//check if we can calculate everything at once or we need to do it in blocks due to memory limitations
		std::int64_t required_mem = full_img_mem.numBytes(view) * 3 + 104857600;//image, weights, temp for convolution + 100MB extra
		std::int64_t availMem = getAvailableMemDeviceCUDA(devCUDA);
		if (availMem > required_mem)
		{
			//computing all at once (most cases)			
			calculateWeightsSingleView_allAtOnce(view, paramDec.anisotropyZ);
		}
		else{
			
			calculateWeightsSingleView_lowMem(view, paramDec.anisotropyZ, availMem);
		}			
	}

}

//=========================================================
void multiGPUblockController::calculateWeightsSingleView_allAtOnce(int view, float anisotropyZ)
{
	//computing all at once (most cases)
	//allocate memory
	full_img_mem.allocateView_GPU(view, full_img_mem.numBytes(view));
	full_img_mem.copyView_CPU_to_GPU(view);
	if (full_weights_mem.getPointer_CPU(view) == NULL)
		full_weights_mem.allocateView_CPU(view, full_img_mem.numBytes(view));

	full_weights_mem.allocateView_GPU(view, full_img_mem.numBytes(view));

	//calculate weights
	calculateWeightsDeconvolution(full_weights_mem.getPointer_GPU(view), full_img_mem.getPointer_GPU(view), full_img_mem.dimsImgVec[view].dims, full_img_mem.dimsImgVec[view].ndims, anisotropyZ);

	//copy weights back
	full_weights_mem.copyView_GPU_to_CPU(view);

	//deallocate memory
	full_img_mem.deallocateView_GPU(view);
	full_weights_mem.deallocateView_GPU(view);
}

//=========================================================
void multiGPUblockController::calculateWeightsSingleView_lowMem(int view, float anisotropyZ, int64_t availMem)
{	
	//allocate final memory for weights in CPU
	if (full_weights_mem.getPointer_CPU(view) == NULL)
		full_weights_mem.allocateView_CPU(view, full_img_mem.numBytes(view));

	//calculate per blocks	
	int64_t stride = 1;//number of pixels on each plane
	for (int ii = 0; ii < full_img_mem.dimsImgVec[view].ndims - 1; ii++)
	{
		stride *= full_img_mem.dimsImgVec[view].dims[ii];
	}	
		
	std::int64_t max_z_dim = (availMem - 104857600) / (3 * stride * sizeof(float));

	cout<<"=====DEBUGGING: maximum number of z planes for dct weight "<<max_z_dim<<" ; available bytes = "<<availMem<<endl;

	//get the padding size
	const int64_t padSize = (ceil( 5.0f * cellDiameterPixels * 0.5 / anisotropyZ ) * 2 + 1 ) / 2;//number of z planes below and above that need ot be disregarded
	const int64_t useful_number_planes = max_z_dim - 2 * padSize;
	if (useful_number_planes < 1) //minimum number of planes to extract useful information
	{
		std::cout << "ERROR: multiGPUblockController_lowMem::calculateWeightsSingleView: not enough memory to calculate DCT even with blocks" << std::endl;
		exit(3);
	}


	//variables (copying from block partition for multiview deconvolution)
	int dimBlockParition = full_img_mem.dimsImgVec[view].ndims - 1;//partition in blocks along the las dimension
	const int64_t PSFpadding = padSize;//quantity that we have to pad on each side to avoid edge effects
	const int64_t chunkSize = std::min(useful_number_planes, int64_t(full_img_mem.dimsImgVec[view].dims[dimBlockParition]));//effective size where we are calculating LR	

	int64_t JoffsetIni = 0, JoffsetEnd;//useful slices in J (global final outout) are from [JoffsetIni,JoffsetEnd)
	int64_t BoffsetIni;//useful slices in ROI (local input) are from [BoffsetIni, BoffsetEnd]. Thus, at the end we copy J(:,.., JoffsetIni:JoffsetEnd-1,:,..:) = Jobj->J(:,.., BoffsetIni:BoffsetEnd-1,:,..:)
	
	uint32_t xyzct[KLB_DATA_DIMS];


	//copy value to define klb ROI
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		xyzct[ii] = full_img_mem.dimsImgVec[view].dims[ii];
	for (int ii = MAX_DATA_DIMS; ii < KLB_DATA_DIMS; ii++)
		xyzct[ii] = 1;

	


	//allocate memory for blocks in the GPU
	float* block_weights_mem_GPU = allocateMem_GPU<float>( stride * max_z_dim);
	float* block_img_mem_GPU = allocateMem_GPU<float>(stride * max_z_dim);
	int64_t blockDims[MAX_DATA_DIMS];
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		blockDims[ii] = xyzct[ii];


	//main loop
	while (JoffsetIni < xyzct[dimBlockParition])
	{		
		klb_ROI ROI;
		ROI.defineSlice(JoffsetIni, dimBlockParition, xyzct);//this would be a slice through dimsBlockParitiondimension equal to offset
		BoffsetIni = PSFpadding;
		if (JoffsetIni >= PSFpadding)//we cannot gain anything
		{
			ROI.xyzctLB[dimBlockParition] -= PSFpadding;
		}
		else{//we can process more slices in the block since we are at the beggining
			ROI.xyzctLB[dimBlockParition] = 0;
			BoffsetIni -= (PSFpadding - JoffsetIni);
		}

		JoffsetEnd = JoffsetIni + chunkSize + PSFpadding - BoffsetIni;
		JoffsetEnd = std::min((uint32_t)JoffsetEnd, xyzct[dimBlockParition]);//make sure we do not go over the end of the image

		
		ROI.xyzctUB[dimBlockParition] = JoffsetEnd + PSFpadding - 1;
		ROI.xyzctUB[dimBlockParition] = std::min(ROI.xyzctUB[dimBlockParition], xyzct[dimBlockParition] - 1);//make sure we do not go over the end of the image	
		

		//copy image and weights ROI to CUDA
		int64_t offset = ROI.xyzctLB[dimBlockParition];
		offset *= stride;
		float* auxPtr_CPU = full_img_mem.getPointer_CPU(view);
		auxPtr_CPU = &(auxPtr_CPU[offset]);
		copy_CPU_to_GPU(auxPtr_CPU, block_img_mem_GPU, ROI.getSizePixels());

		auxPtr_CPU = full_weights_mem.getPointer_CPU(view);
		auxPtr_CPU = &(auxPtr_CPU[offset]);
		copy_CPU_to_GPU(auxPtr_CPU, block_weights_mem_GPU, ROI.getSizePixels());
		

		//calculate weights
		blockDims[dimBlockParition] = ROI.getSizePixels(dimBlockParition);
		calculateWeightsDeconvolution(block_weights_mem_GPU, block_img_mem_GPU, blockDims, full_img_mem.dimsImgVec[view].ndims, anisotropyZ, false);//we cannot normalize each block independently

		//copy block result to CPU weights
		offset = stride * BoffsetIni;
		float* auxPtr_CUDA = &(block_weights_mem_GPU[offset]);
		auxPtr_CPU = full_weights_mem.getPointer_CPU(view);
		auxPtr_CPU = &(auxPtr_CPU[stride * JoffsetIni]);
		size_t blockSize = stride * (JoffsetEnd - JoffsetIni);
		copy_GPU_to_CPU(auxPtr_CPU, auxPtr_CUDA, blockSize);

		//update offset counter
		JoffsetIni = JoffsetEnd;
	}
	//deallocate GPU memory
	deallocateMem_GPU<float>(block_weights_mem_GPU);
	deallocateMem_GPU<float>(block_img_mem_GPU);		

	//normalize weight in the CPU
	float minW = numeric_limits<float>::max();
	float maxW = -minW;
	float* auxPtr = full_weights_mem.getPointer_CPU(view);
	for (int ii = 0; ii < full_weights_mem.numElements(view); ii++)
	{
		minW = std::min(minW, auxPtr[ii]);
		maxW = std::max(maxW, auxPtr[ii]);
	}
	maxW -= minW;
	for (int ii = 0; ii < full_weights_mem.numElements(view); ii++)
	{
		auxPtr[ii] = (auxPtr[ii] - minW) / maxW;
	}
}

//=========================================================
void multiGPUblockController::copyBlockResultToJ(const imgTypeDeconvolution* Jsrc, const uint32_t blockDims[MAX_DATA_DIMS], int64_t Joffset, int64_t Boffset, int64_t numPlanes)
{
	if (dimBlockParition != 1)
	{
		cout << "TODO:ERROR: copyBlockResultToJ: function hardcoded to stitch in y-direction" << endl;
		exit(3);
	}

	if ( MAX_DATA_DIMS != 3)
	{
		cout << "TODO:ERROR: copyBlockResultToJ: function hardcoded to stitch 3D arrays" << endl;
		exit(3);
	}

	size_t lineSize = blockDims[0] * sizeof(imgTypeDeconvolution);
	int64_t idx = 0;
	int64_t count = 0;
	for (int64_t zz = 0; zz < blockDims[2]; zz++)
	{
		idx = imgDims[0] * (Joffset + imgDims[1] * zz);
		count = blockDims[0] * (Boffset + blockDims[1] * zz);
		for (int64_t yy = Boffset; yy < Boffset + numPlanes; yy++)
		{
			//update for new array
			//idx = dimsAfterPad[0] * ( yy + dimsAfterPad[1] * zz);
			//update for new array
			//count = dimsNow[0] * (yy + dimsNow[1] * zz);

			//copy elements
			memcpy(&(J[idx]), &(Jsrc[count]), lineSize);

			//update counters
			idx += imgDims[0];
			count += blockDims[0];
		}
	}
}

//==========================================================
int multiGPUblockController::getImageDimensions()
{
	string filename = multiviewImage<imgTypeDeconvolution>::recoverFilenamePatternFromString(paramDec.filePatternImg, 1);
	return multiviewImage<imgTypeDeconvolution>::getImageDimensionsFromHeader(filename, imgDims);	
}

//==========================================================
uint64_t multiGPUblockController::numElements()
{
	uint64_t nImg = 1;
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		nImg *= (uint64_t)(imgDims[ii]);

	return nImg;
}


//==========================================================
void multiGPUblockController::debug_listGPUs()
{
	for (const auto &p : GPUinfoVec)
	{
		cout << "Dev CUDA " << p.devCUDA << ": " << p.devName << " with mem = " << p.mem << " bytes" << endl;
	}
}
//=============================================
//===========================================================================================
int multiGPUblockController::writeDeconvoutionResult(const std::string& filename)
{
	//initialize I/O object	
	klb_imageIO imgIO(filename);

	uint32_t xyzct[KLB_DATA_DIMS];
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{
		xyzct[ii] = imgDims[ii];
	}
	for (int ii = MAX_DATA_DIMS; ii <KLB_DATA_DIMS; ii++)
	{
		xyzct[ii] = 1;
	}

	//set header
	switch (sizeof(imgTypeDeconvolution))//TODO: this is not accurate since int8 will be written as uint8
	{
	case 1:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT8_TYPE);
		break;
	case 2:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT16_TYPE);
		break;
	case 4:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::FLOAT32_TYPE);
		break;
	default:
		cout << "ERROR: format not supported yet" << endl;
		return 10;
	}
	
	//write image
	int error = imgIO.writeImage((char*)(J), -1);//all the threads available

	if (error > 0)
	{
		switch (error)
		{
		case 2:
			printf("Error during BZIP compression of one of the blocks");
			break;
		case 5:
			printf("Error generating the output file in the specified location");
			break;
		default:
			printf("Error writing the image");
		}
	}

	return error;
}
//=============================================

//===========================================================================================
int multiGPUblockController::writeDeconvoutionResult_uint16(const std::string& filename)
{
	int error;

	//initialize I/O object	
	klb_imageIO imgIO(filename);

	uint32_t xyzct[KLB_DATA_DIMS];
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{
		xyzct[ii] = imgDims[ii];
	}
	for (int ii = MAX_DATA_DIMS; ii <KLB_DATA_DIMS; ii++)
	{
		xyzct[ii] = 1;
	}

	//set header
	switch (sizeof(imgTypeDeconvolution))//TODO: this is not accurate since int8 will be written as uint8
	{
	case 1:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT8_TYPE);		
		error = imgIO.writeImage((char*)(J), -1);//all the threads available
		break;
	case 2:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT16_TYPE);
		error = imgIO.writeImage((char*)(J), -1);//all the threads available
		break;
	case 4:
	{	//parse data to uint16
			  imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT16_TYPE);

			  uint64_t N = numElements();
			  uint16_t *Jaux = new uint16_t[N];
			  float Imin = 1e32, Imax = -1e32;
			  for (uint64_t ii = 0; ii < N; ii++)
			  {
				  Imin = min(J[ii], Imin);
				  Imax = max(J[ii], Imax);
			  }
			  Imax = Imax - Imin;
			  for (uint64_t ii = 0; ii < N; ii++)
			  {
				  Jaux[ii] = (uint16_t)(4096.0f * (J[ii] - Imin) / Imax);//we do not need the whole uint16 dynamic range and it helps compression
			  }

			  error = imgIO.writeImage((char*)(Jaux), -1);//all the threads available

			  delete[] Jaux;
			  break;
	}
	default:
		cout << "ERROR: format not supported yet" << endl;
		return 10;
	}

	

	if (error > 0)
	{
		switch (error)
		{
		case 2:
			printf("Error during BZIP compression of one of the blocks");
			break;
		case 5:
			printf("Error generating the output file in the specified location");
			break;
		default:
			printf("Error writing the image");
		}
	}

	return error;
}
//=============================================
//===========================================================================================
int multiGPUblockController::writeDeconvoutionResultRaw(const std::string& filename)
{
	
	FILE* fid = fopen(filename.c_str(), "wb");

	if (fid == NULL)
	{
		printf("Error opening file %s to save raw image data\n", filename.c_str());
		return 2;
	}

	fwrite((void*)(J), sizeof(imgTypeDeconvolution), numElements(), fid);
	fclose(fid);

	//write header information
	string filenameH(filename + ".txt");
	fid = fopen(filenameH.c_str(), "w");
	if (fid == NULL)
	{
		printf("Error opening file %s to save header\n", filenameH.c_str());
		return 2;
	}

	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{
		fprintf(fid, "%d ", (int)(imgDims[ii]));
	}
	fprintf(fid, "\n");

	switch (sizeof(imgTypeDeconvolution))
	{
	case 1:
		fprintf(fid, "uint8\n");
		break;
	case 2:
		fprintf(fid, "uint16\n");
		break;
	case 4:
		fprintf(fid, "single\n");
		break;
	default:
		fprintf(fid, "unkown\n");
		break;
	}


	fclose(fid);
	return 0;
}
