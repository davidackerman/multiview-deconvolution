/*
*
* Authors: Fernando Amat
*  test_multiview_deconvolution_LR_multiGPU.cpp.cpp
*
*  Created on : June 11th, 2015
* Author : Fernando Amat
*
* \brief testing a full iteration of multi-view lucy richardson splitting image in blocks and multiple GPUS
*
*/

#include <iostream>
#include <cstdint>
#include <time.h>       /* time_t, struct tm, difftime, time, mktime */
#include "multiGPUblockController.h"


using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "testing a full iteration of multi-view lucy richardson splitting it in blocks and multiple GPUs running..." << std::endl;
	time_t start, end;

	time(&start);
	//parameters
	string filepath("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/");
	int numIters = 2;
	int numViews = 4;
	imgTypeDeconvolution imgBackground = 100;
	cout << "===============TODO: activate total variations==============" << endl;
	float lambdaTV = -1.0;//0.008;
	string filePatternPSF( filepath + "psfReg_?.klb");
	string filePatternWeights(filepath + "weightsReg_?.klb");
	string filePatternImg(filepath + "imReg_?.klb");


	if (argc > 1)
		filepath = string(argv[1]);
	if (argc > 2)
		numIters = atoi(argv[2]);

    //main object to control the process
	multiGPUblockController master;

    //set paramaters
	master.paramDec.filepath = filepath;
	master.paramDec.filePatternImg = filePatternImg;
	master.paramDec.filePatternPSF = filePatternPSF;
	master.paramDec.filePatternWeights = filePatternWeights;
	master.paramDec.lambdaTV = lambdaTV;
	master.paramDec.imgBackground = imgBackground;
	master.paramDec.numIters = numIters;
	master.paramDec.Nviews = numViews;

    //check number of GPUs and the memory available for each of them
	master.queryGPUs();

#ifdef _DEBUG
	master.debug_listGPUs();
#endif

    //find out best dimension to perform blocks for minimal padding
	int err = master.findBestBlockPartitionDimension();
	if (err > 0)
		return err;

    //precalculate number of planes per GPU we can do (including padding to avoid border effect)
	master.findMaxBlockPartitionDimensionPerGPU();

	cout << "==============WARNING: manually modiying findMaxBlockPartitionDimensionPerGPU value to test with two GPUs==================" << endl;
	for (size_t ii = 0; ii < master.getNumGPU(); ii++)
		master.debug_setGPUmaxSizeDimBlockPartition(ii, 75);

    //launch multi-thread as a producer consumer queue to calculate blocks as they come
	err = master.runMultiviewDeconvoution();
	if (err > 0)
		return err;

	time(&end);
	cout << "Multiview deconvolution using multi-GPU took " << difftime(end, start) << " secs for " << numIters << " iterations" << endl;
    //write result
	err = master.writeDeconvoutionResult(string(filepath + "test_mv_deconv_LR_multiGPU.klb"));
	if (err > 0)
	{
		cout << "ERROR: writing result" << endl;
		return err;
	}

	return 0;
}