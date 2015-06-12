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
#include "multiGPUblockController.h"
#include "standardCUDAfunctions.h"
#include "klb_ROI.h"
#include "klb_imageIO.h"
#include "klb_Cwrapper.h"







using namespace std;

multiGPUblockController::multiGPUblockController()
{
	dimBlockParition = -1;
	J = NULL;
}
//============================================================
multiGPUblockController::~multiGPUblockController()
{
	if (J != NULL)
		delete[] J;
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

}


//================================================================
int multiGPUblockController::runMultiviewDeconvoution()
{
    //calculate image size	
	int err = getImageDimensions();
	if (err > 0)
		return err;

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
		threads.push_back(std::thread(&multiGPUblockController::multiviewDeconvolutionBlockWise, this, ii));		
	}

	//wait for the workers to finish
	for (auto& t : threads)
		t.join();

	return 0;
}


//==========================================================
void multiGPUblockController::multiviewDeconvolutionBlockWise(size_t threadIdx)
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
		cout << "WARNING: device CUDA " << GPUinfoVec[threadIdx].devCUDA << " with GPU " << GPUinfoVec[threadIdx].devName<<" cannot provess enough effective planes after padding" << endl;
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
		Jobj->deconvolution_LR_TV(paramDec.numIters, paramDec.lambdaTV);

        //copy block result to J
		Jobj->copyDeconvoutionResultToCPU();
		copyBlockResultToJ(Jobj->getJpointer(), blockDims, JoffsetIni, BoffsetIni, JoffsetEnd - JoffsetIni);
	}


	delete Jobj;
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