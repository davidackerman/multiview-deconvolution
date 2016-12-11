/*
*
* Authors: Fernando Amat
*  main_multiviewDeconvLR_multiGPU_low_mem.cxx
*
*  Created on : July 27th, 2015
* Author : Fernando Amat
*
* \brief main executable to perform multiview deconvolution Lucy-Richardson with multi-GPU using temporary saved files on disk to save memory (for large images or computers with low memory)
*
*/

#include <iostream>
#include <sstream>
#include <iomanip>  
#include <string>
#include <chrono>
#include <cstdio>
#include <math.h>  
#include <algorithm>
#include "multiGPUblockController.h"
#include "klb_Cwrapper.h"
#include "imgUtils.h"


using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, const char** argv)
{
	auto tStart = Clock::now();
	auto t1 = Clock::now();
	auto t2 = Clock::now();

	//main inputs
	string filenameXML("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/reg_deconv/regDeconvParam.xml");
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
	//check number of GPUs and the memory available for each of them
	master.queryGPUs(maxNumberGPU);
	master.debug_listGPUs();
		
	for (int ii = 0; ii < master.paramDec.Nviews; ii++)
	{
		t1 = Clock::now();
		cout << "Reading view " << ii << endl;
		master.full_img_mem.readImage(master.paramDec.fileImg[ii], -1);		

		cout << "Reading PSF for view " << ii << endl;
		master.full_psf_mem.readImage(master.paramDec.filePSF[ii], -1);

		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	}

    // Do affine transform of PSFs, if needed.  We do this now because things like master.findBestBlockPartitionDimension_inMem()
    // assume that the PSFs are already transformed
    for (int ii = 0; ii < master.paramDec.Nviews; ii++)
    {
        if (master.paramDec.isPSFAlreadyTransformed)
        {
            cout << "PSF is already transformed, so not transforming further " << endl;
        }
        else
        {
            cout << "Applying affine transformation to PSF " << endl;
            master.full_psf_mem.apply_affine_transformation_psf(ii, &(master.paramDec.AcellAsDouble[ii][0]), 3);  
              // 3 as third argument means cubic interpolation, with points outside the stack assumed to be zero
        }
    }

	t1 = Clock::now();
	cout << "Calculating constrast weights for each view in GPU" << endl;
	master.calculateWeights();
	t2 = Clock::now();
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	
	int64_t dimsOut[3];
	const int refView = 0;
	cout << "We assume first view is the reference. So its affine transformation is just a scaling in Z" << endl;
	//set parameters	
	dimsOut[0] = master.full_img_mem.dimsImgVec[refView].dims[0];
	dimsOut[1] = master.full_img_mem.dimsImgVec[refView].dims[1];
	dimsOut[2] = ceil(master.full_img_mem.dimsImgVec[refView].dims[2] * master.paramDec.anisotropyZ);

	//find out best dimension to perform blocks for minimal padding
	int err = master.findBestBlockPartitionDimension_inMem();
	if (err > 0)
		return err;

	//adjust dimensions so they are good for FFT (factors of 2 and 3)
	for (int ii = 0; ii < 3; ii++)
	{
		if (master.getDimBlockPartition() != ii)
			dimsOut[ii] = master.padToGoodFFTsize(dimsOut[ii]);
	}

	//apply transformation to weights and then save it in disk to reduce memory consumption
	int64_t imSize = dimsOut[0] * dimsOut[1] * dimsOut[2];
	uint16_t *weightsCompress = new uint16_t[imSize];
	uint32_t xyzct[KLB_DATA_DIMS] = {dimsOut[0], dimsOut[1], dimsOut[2], 1, 1};
	for (int ii = 0; ii < master.paramDec.Nviews; ii++)
	{		
		t1 = Clock::now();
		cout << "Applying affine transformation to weight array " << endl;
		master.full_weights_mem.apply_affine_transformation_img(ii, dimsOut, &(master.paramDec.Acell[ii][0]), 1);//linear interpolation with border pixels assigned to 0
		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

		t1 = Clock::now();
		std::string tmpFile = generateTempFilename("MVddec_");//temporary filename	
		tmpFile += ".klb";
		cout << "Saving weight array in temporary file "<<tmpFile << endl;
		weightType* wPtr = master.full_weights_mem.getPointer_CPU(ii);
		for (int64_t jj = 0; jj < imSize; jj++)
		{
			weightsCompress[jj] = (uint16_t)(100 * wPtr[jj]);//this is plenty quantization for weights and allows KLB to compress the file by a lot
		}
		err = writeKLBstack((void*)weightsCompress, tmpFile.c_str(), xyzct, KLB_DATA_TYPE::UINT16_TYPE, -1, NULL, NULL, KLB_COMPRESSION_TYPE::BZIP2, NULL);
		if (err > 0)
			return err;
		master.full_weights_filename.push_back(tmpFile);
		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
		//deallocate weight memory
		master.full_weights_mem.deallocateView_CPU(ii);
	}
	delete[] weightsCompress;

	//apply transofrmation to images (in this case we save it in memory)
	for (int ii = 0; ii < master.paramDec.Nviews; ii++)
	{
		t1 = Clock::now();
		cout << "Applying affine transformation to view " << ii << endl;
		master.full_img_mem.apply_affine_transformation_img(ii, dimsOut, &(master.paramDec.Acell[ii][0]), 3);//cubic interpolation with border pixels assigned to 0
		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;		
	}


	if (master.paramDec.verbose > 0)
	{
		cout << "Saving transformed images and PSFs for all views in file " << filenameXML <<"*"<<endl;
		for (int ii = 0; ii < master.paramDec.Nviews; ii++)
		{
            string debugStackFileName = filenameXML + "_debug_img_" + to_string(ii) + ".klb" ;
            master.full_img_mem.writeImage_uint16(debugStackFileName, ii, 4096.0f);
            string debugPSFFileName = filenameXML + "_debug_psf_" + to_string(ii) + ".klb" ;
            master.full_psf_mem.writeImage(debugPSFFileName, ii);
        }
	}

	//precalculate number of planes per GPU we can do (including padding to avoid border effect)
	master.findMaxBlockPartitionDimensionPerGPU_inMem();

	t1 = Clock::now();
	cout << "Calculating multiview deconvolution..." << endl;
	//launch multi-thread as a producer consumer queue to calculate blocks as they come
	err = master.runMultiviewDeconvoution(&multiGPUblockController::multiviewDeconvolutionBlockWise_lowMem);
	if (err > 0)
		return err;
	t2 = Clock::now();
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	

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
	//sprintf(fileoutName, "%s_dec_LR_multiGPU_%s_iter%d_lambdaTV%.6d", master.paramDec.fileImg[0].c_str(), master.paramDec.outputFilePrefix.c_str(), master.paramDec.numIters, (int)(1e6f * std::max(master.paramDec.lambdaTV, 0.0f)));
	t1 = Clock::now();
    cout << "Writing result to " << outputFileName << endl;
	if (master.paramDec.saveAsUINT16)
        err = master.writeDeconvoutionResult_uint16(outputFileName);
	else
        err = master.writeDeconvolutionResult_float(outputFileName);

	if (err > 0)
	{
		cout << "ERROR: writing result" << endl;
		return err;
	}
	t2 = Clock::now();
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;	
	

	//delete temporary files
	cout << "Deleting temporary files " << endl;
	for (int ii = 0; ii < master.paramDec.Nviews; ii++)
	{
		remove(master.full_weights_filename[ii].c_str());
	}


	auto tEnd = Clock::now();
	std::cout << "Total process  took " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count() << " ms" << std::endl;

	return 0;
}
