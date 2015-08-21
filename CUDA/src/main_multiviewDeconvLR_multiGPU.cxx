/*
*
* Authors: Fernando Amat
*  main_multiviewDeconvLR_multiGPU.cxx
*
*  Created on : July 27th, 2015
* Author : Fernando Amat
*
* \brief main executable to perform multiview deconvolution Lucy-Richardson with multi-GPU
*
*/

#include <iostream>
#include <string>
#include <chrono>
#include <math.h>  
#include <algorithm>
#include "multiGPUblockController.h"


using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, const char** argv)
{
	auto tStart = Clock::now();
	auto t1 = Clock::now();
	auto t2 = Clock::now();

	//main inputs
	string filenameXML("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/reg_deconv_bin4/regDeconvParam.xml");
	int maxNumberGPU = -1;//default value

	if (argc > 1)
		filenameXML = string(argv[1]);
	if (argc > 2)
		maxNumberGPU = atoi(argv[2]);

	//--------------------------------------------------------
	//main object to control the deconvolution process
	cout << "Reading parameters from XML file" << endl;
	multiGPUblockController master(filenameXML);

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


	t1 = Clock::now();
	cout << "Calculating constrast weights for each view in GPU" << endl;
	master.calculateWeights();
	t2 = Clock::now();
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	
	int64_t dimsOut[3];
	const int refView = 0;
	cout << "We assume first view is the reference. So its affine transformation is just a scalign in Z" << endl;
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

	//apply transformation
	for (int ii = 0; ii < master.paramDec.Nviews; ii++)
	{
				
		t1 = Clock::now();
		cout << "Applying affine transformation to view " << ii << endl;
		master.full_img_mem.apply_affine_transformation_img(ii, dimsOut, &(master.paramDec.Acell[ii][0]), 3);//cubic interpolation with border pixels assigned to 0
		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

		t1 = Clock::now();
		cout << "Applying affine transformation to weight array " << endl;
		master.full_weights_mem.apply_affine_transformation_img(ii, dimsOut, &(master.paramDec.Acell[ii][0]), 1);//linear interpolation with border pixels assigned to 0
		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

	}

	if (master.paramDec.verbose > 0)
	{
		cout << "Saving transformed images and weights for all views in file " << filenameXML <<"*"<<endl;
		for (int ii = 0; ii < master.paramDec.Nviews; ii++)
		{
			char buffer[256];
			sprintf(buffer, "%s_debug_img_%d.klb", filenameXML.c_str(), ii);
			master.full_img_mem.writeImage_uint16(string(buffer), ii, 4096.0f);
			sprintf(buffer, "%s_debug_weigths_%d.klb", filenameXML.c_str(), ii);
			master.full_weights_mem.writeImage_uint16(string(buffer), ii, 100);
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
	

	//write result
	char fileoutName[256];	
	sprintf(fileoutName, "%s_dec_LR_multiGPU_%s_iter%d_lambdaTV%.6d.klb", master.paramDec.fileImg[0].c_str(), master.paramDec.outputFilePrefix.c_str(), master.paramDec.numIters, (int)(1e6f * std::max(master.paramDec.lambdaTV, 0.0f)));
	t1 = Clock::now();
	cout << "Writing result to "<<string(fileoutName) << endl;
	//err = master.writeDeconvoutionResult(string(fileoutName)));
	err = master.writeDeconvoutionResult_uint16(string(fileoutName));
	if (err > 0)
	{
		cout << "ERROR: writing result" << endl;
		return err;
	}
	t2 = Clock::now();
	std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;	

	auto tEnd = Clock::now();
	std::cout << "Total process  took " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count() << " ms" << std::endl;

	return 0;
}
