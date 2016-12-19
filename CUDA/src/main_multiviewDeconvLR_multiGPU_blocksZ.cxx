/*
*
* Authors: Fernando Amat
*  main_multiviewDeconvLR_multiGPU_blocksZ.cxx
*
*  Created on : July 27th, 2015
* Author : Fernando Amat
*
* \brief main executable to perform multiview deconvolution Lucy-Richardson with multi-GPU
*
*/

#include <iostream>
#include <sstream>
#include <iomanip>  
#include <string>
#include <chrono>
#include <math.h> 
#include <algorithm>
#include "multiGPUblockController.h"
#include "klb_ROI.h"
#include "standardCUDAfunctions.h"

using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, const char** argv)
{
    int err = 0 ;
	auto tStart = Clock::now();
	auto t1 = Clock::now();
	auto t2 = Clock::now();

	//main inputs
	string filenameXML("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/reg_deconv_bin4/regDeconvParam.xml");
	int maxNumberGPU = -1;//default value
    bool wasOutputFileNameGiven = false ;
    string outputFileName("") ;

	if (argc > 1)
		filenameXML = string(argv[1]);
	if (argc > 2)
		maxNumberGPU = atoi(argv[2]);
    if (argc > 3)
    {
        outputFileName = string(argv[3]);
        wasOutputFileNameGiven = true ;
    }

	//--------------------------------------------------------
	//main object to control the deconvolution process
	cout << "Reading parameters from XML file" << endl;
	multiGPUblockController master(filenameXML);
	
	master.setWeightsThreshold(master.paramDec.weightThr);

	if (master.paramDec.blockZsize <= 0)
	{
		cout << "ERROR: blockSize is negative. Use main_multiviewDeconvLR_multiGPU for better performance" << endl;
		return 2;
	}

	//make sure block size in Z is conveninet for FFT
	master.paramDec.blockZsize = master.ceilToGoodFFTsize(master.paramDec.blockZsize);

	//check number of GPUs and the memory available for each of them
	master.queryGPUs(maxNumberGPU);
	master.debug_listGPUs();

	//read first image to calculate output dimensions
	int64_t dimsOut[3];
	const int refView = 0;
	cout << "We assume view " << refView << " is the reference. So its affine transformation is just a scaling in Z" << endl;
	//set final dimensions of the stack
	master.full_img_mem.readImageSizeFromHeader(master.paramDec.fileImg[refView], dimsOut);
	dimsOut[2] = ceil(dimsOut[2] * master.paramDec.anisotropyZ);

	//read PSF to calculate padding in Z
	int64_t maxPSFblockSize_Z = 0;
	for (int ii = 0; ii < master.paramDec.Nviews; ii++)
	{
		cout << "Reading PSF for view " << ii << endl;
		err = master.full_psf_mem.readImage(master.paramDec.filePSF[ii], -1);
        if (err) return err ;
        // Transform the PSF, if called for
        if (master.paramDec.isPSFAlreadyTransformed)
        {
            cout << "PSF is already transformed, so not transforming further " << endl;
        }
        else
        {
            cout << "Applying affine transformation to PSF" << endl;
            master.full_psf_mem.apply_affine_transformation_psf(ii, &(master.paramDec.AcellAsDouble[ii][0]), 3);//cubic interpolation with border pixels assigned to 0
        }
		//calculate padding in z
		maxPSFblockSize_Z = std::max(maxPSFblockSize_Z, master.full_psf_mem.dimsImgVec[ii].dims[2]);//to avoid needing to set NOMINMAX window define
		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	}

	//find out best dimension to perform blocks for minimal padding
	err = master.findBestBlockPartitionDimension_inMem();
	if (err > 0)
		return err;

	//adjust dimensions so they are good for FFT (factors of 2 and 3)
	int64_t imRefSize = 1;
	for (int ii = 0; ii < 3; ii++)
	{
		if (master.getDimBlockPartition() != ii)
			dimsOut[ii] = master.padToGoodFFTsize(dimsOut[ii]);

		imRefSize *= dimsOut[ii];
	}

	//find out information for blocks in Z
	const int64_t PSFpadding = 1 + maxPSFblockSize_Z / 2;//quantity that we have to pad on each side to avoid edge effects
	const int64_t chunkSize = std::min((int64_t)(master.paramDec.blockZsize) - 2 * PSFpadding, dimsOut[2]);//effective size where we are calculating LR	

	if (chunkSize <= 0)
	{
		cout << "ERROR: chunksize (" << chunkSize << ") is negative. Please increase the blockZsize attribute or reduce PSF size on Z" << endl;
		return 3;
	}
	cout << "Maximum PSF block size in Z is " << maxPSFblockSize_Z << " pixels. Code called with Z blocks of size " << master.paramDec.blockZsize << " pixels. Effective number of planes per iteration is " << chunkSize << endl;


	//variables to keep track of copies between block and final result
	imgTypeDeconvolution* JfullImg = new imgTypeDeconvolution[imRefSize];
	int64_t JoffsetIni = 0, JoffsetEnd;//useful slices in J (global final outout) are from [JoffsetIni,JoffsetEnd)
	int64_t BoffsetIni;//useful slices in ROI (local input) are from [BoffsetIni, BoffsetEnd]. Thus, at the end we copy J(:,.., JoffsetIni:JoffsetEnd-1,:,..:) = Jobj->J(:,.., BoffsetIni:BoffsetEnd-1,:,..:)
	const int dimBlockParition = 2;//along z
	const int64_t stride = dimsOut[0] * dimsOut[1];//number of pixels on each plane

	uint32_t xyzct[KLB_DATA_DIMS];
	//copy value to define klb ROI
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		xyzct[ii] = dimsOut[ii];
	for (int ii = MAX_DATA_DIMS; ii < KLB_DATA_DIMS; ii++)
		xyzct[ii] = 1;

	uint32_t blockDims[MAX_DATA_DIMS];//maximum size of a block to preallocate memory
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		blockDims[ii] = xyzct[ii];//image size except a long the dimension of partition
	blockDims[dimBlockParition] = std::min(master.paramDec.blockZsize, (int)(dimsOut[dimBlockParition]));

	//main loop
	int iter = 0;
	while (JoffsetIni < dimsOut[dimBlockParition])
	{
		cout << "Processing block in Z with starting useful plane = " << JoffsetIni << endl;

		//calculate subblock
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
				

		//apply transformation
		for (int ii = 0; ii < master.paramDec.Nviews; ii++)
		{
			t1 = Clock::now();
			cout << "Reading view " << ii << endl;
			err = master.full_img_mem.readImage(master.paramDec.fileImg[ii], -1);
            if (err) return err ;
            t2 = Clock::now();
			std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;			

			t1 = Clock::now();
			cout << "Calculating constrast weights for view "<<ii<< " in GPU" << endl;
			master.calculateWeights(ii,master.getDevCUDA_maxMem());
			t2 = Clock::now();
			std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

			t1 = Clock::now();
			cout << "Applying affine transformation to view " << ii << endl;
			master.full_img_mem.apply_affine_transformation_img(ii, dimsOut, &(master.paramDec.Acell[ii][0]), 3);//cubic interpolation with border pixels assigned to 0
			t2 = Clock::now();
			std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

			//crop image to correct block in Z			
			int64_t offset = ROI.xyzctLB[dimBlockParition];
			offset *= stride;
			float* auxPtr_CPU = master.full_img_mem.getPointer_CPU(ii);
			auxPtr_CPU = &(auxPtr_CPU[offset]);
			outputType* imgBlockptr = new outputType[ROI.getSizePixels()];
			memcpy(imgBlockptr, auxPtr_CPU, ROI.getSizePixels() * sizeof(outputType));
			master.full_img_mem.deallocateView_CPU(ii);//deallocate full views
			master.full_img_mem.dimsImgVec[ii].dims[dimBlockParition] = ROI.getSizePixels(dimBlockParition);//redefine dimensions
			master.full_img_mem.setPointer_CPU(ii, imgBlockptr);//set new pointer

			t1 = Clock::now();
			cout << "Applying affine transformation to weight array " << endl;
			master.full_weights_mem.apply_affine_transformation_img(ii, dimsOut, &(master.paramDec.Acell[ii][0]), 1);//linear interpolation with border pixels assigned to 0
			t2 = Clock::now();
			std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;


			//crop image to correct block in Z						
			auxPtr_CPU = master.full_weights_mem.getPointer_CPU(ii);
			auxPtr_CPU = &(auxPtr_CPU[offset]);
			imgBlockptr = new outputType[ROI.getSizePixels()];
			memcpy(imgBlockptr, auxPtr_CPU, ROI.getSizePixels() * sizeof(outputType));
			master.full_weights_mem.deallocateView_CPU(ii);//deallocate full views
			master.full_weights_mem.dimsImgVec[ii].dims[dimBlockParition] = ROI.getSizePixels(dimBlockParition);//redefine dimensions
			master.full_weights_mem.setPointer_CPU(ii, imgBlockptr);//set new pointer


			//the last block has to be padded at the end to match the block dimensions
			if (ROI.getSizePixels(dimBlockParition) != blockDims[dimBlockParition])
			{
				master.full_img_mem.padArrayWithZeros(ii, blockDims);
				master.full_weights_mem.padArrayWithZeros(ii, blockDims);
			}

		}


		if (master.paramDec.verbose > 0)
		{
			cout << "Saving transformed images, weights, and PSFs for all views in file " << filenameXML << "*" << endl;
			for (int ii = 0; ii < master.paramDec.Nviews; ii++)
			{
				char buffer[256];
				sprintf(buffer, "%s_debug_img_%d_iter_%.2d.klb", filenameXML.c_str(), ii, iter);
				master.full_img_mem.writeImage_uint16(string(buffer), ii, 4096.0f);
				sprintf(buffer, "%s_debug_weights_%d_iter_%.2d.klb", filenameXML.c_str(), ii, iter);
				master.full_weights_mem.writeImage_uint16(string(buffer), ii, 100);
                string debugPSFFileName = filenameXML + "_debug_psf_" + to_string(ii) + ".klb" ;
                master.full_psf_mem.writeImage(debugPSFFileName, ii);
			}
		}

		//precalculate number of planes per GPU we can do (including padding to avoid border effect)
		master.findMaxBlockPartitionDimensionPerGPU_inMem();

		t1 = Clock::now();
		cout << "Calculating multiview deconvolution..." << endl;
		//launch multi-thread as a producer consumer queue to calculate blocks as they come
		err = master.runMultiviewDeconvoution(&multiGPUblockController::multiviewDeconvolutionBlockWise_fromMem);
		if (err > 0)
			return err;
		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;


		//copy block deconvolution result to final result		
		int64_t offset = stride * BoffsetIni;
		float* auxPtr_Jblock = master.getJpointer();
		auxPtr_Jblock = &(auxPtr_Jblock[offset]);				
		size_t blockSize = stride * (JoffsetEnd - JoffsetIni);
		memcpy(&(JfullImg[stride * JoffsetIni]), auxPtr_Jblock, blockSize * sizeof(imgTypeDeconvolution));
	
		//release temporary memory
		master.deallocateMemJ();
		master.full_img_mem.clear();
		master.full_weights_mem.clear();
		//update offset counter
		JoffsetIni = JoffsetEnd;
		iter++;
	}

	//write result
    if (!wasOutputFileNameGiven)
    {
        int lambdaAsInt(1e6f * std::max(master.paramDec.lambdaTV, 0.0f)) ;
        stringstream lambdaAsStringStream;
        lambdaAsStringStream << setw(6) << setfill('0') << lambdaAsInt ;
        outputFileName = 
            master.paramDec.fileImg[0] + 
            "_dec_LR_multiGPU_" + 
            master.paramDec.outputFilePrefix + 
            "_iter" + 
            to_string(master.paramDec.numIters) + 
            "_lambdaTV" + 
            lambdaAsStringStream.str() + 
            ".klb" ;
    }
	//char fileoutName[256];
	//sprintf(fileoutName, "%s_dec_LR_multiGPU_%s_iter%d_lambdaTV%.6d.klb", master.paramDec.fileImg[0].c_str(), master.paramDec.outputFilePrefix.c_str(), master.paramDec.numIters, (int)(1e6f * std::max(master.paramDec.lambdaTV, 0.0f)));
	t1 = Clock::now();
    cout << "Writing result to " << outputFileName << endl;
	if (master.paramDec.saveAsUINT16)
        err = master.writeDeconvoutionResult_uint16(outputFileName, JfullImg, xyzct);
	else
        err = master.writeDeconvolutionResult_float(outputFileName, JfullImg, xyzct);
	if (err > 0)
	{
		cout << "ERROR: writing result" << endl;
		return err;
	}
	t2 = Clock::now();
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;	

	//release memory
	delete[] JfullImg;

#ifdef _DEBUG
	cout << "====DEBUGGING: reseting GPUs to allow memory leak checks======" << endl;
	for (size_t ii = 0; ii < master.getNumGPU(); ii++)
		resetDeviceCUDA(ii);
#endif

	auto tEnd = Clock::now();
	std::cout << "Total process  took " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count() << " ms" << std::endl;

	return 0;
}
