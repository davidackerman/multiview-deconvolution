/*
*
* Authors: Fernando Amat
*  test_gpu_elementwiseOp.cpp
*
*  Created on : June 5th, 2015
* Author : Fernando Amat
*
* \brief testing GPU kernels to perform pointwise operations
*
*/

#include <cstdint>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <algorithm>
#include <fstream>
#include "cuda.h"
#include "book.h"
#include "multiviewDeconvolution.h"


typedef float dataType;

using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "testing GPU multi-view deconvolution weights calculation in the GPU running..." << std::endl;

	int devCUDA = 0;
	
	//parameters
	string filepath("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/");
	if (argc > 1)
		filepath = string(argv[1]);
	

	
	string filePatternImg(filepath + "imReg_?.klb");
	string filePatternWeights(filepath + "weightsReg_?.klb");
	int numViews = 1;
	float anisotropyZ = 5.0;

	//=====================================================================

	HANDLE_ERROR(cudaSetDevice(devCUDA));
	

	//declare object	
	multiviewDeconvolution<float> J;
	J.setNumberOfViews(numViews);
	
	//read images
	string filename;
	int err;
	for (int ii = 0; ii < numViews; ii++)
	{
		//calculate weights
		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternImg, ii + 1);
		err = J.readImage(filename, ii, std::string("img"));
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}

		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternWeights, ii + 1);
		err = J.readImage(filename, ii, std::string("weight"));//this function should just read image
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}

		//calculate weights
		J.calculateWeights(ii, anisotropyZ); 		

		//compare weights
		char buffer[256];
		sprintf(buffer, "%sdebug_weightsRef_%d.raw", filepath.c_str(), ii + 1);
		J.debug_writeGPUarray_weights(ii, string(buffer));
	}

	

	return 0;
}
