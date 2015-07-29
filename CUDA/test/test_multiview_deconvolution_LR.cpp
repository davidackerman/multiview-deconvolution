/*
*
* Authors: Fernando Amat
*  test_load_multiview_images.cpp
*
*  Created on : June 5th, 2015
* Author : Fernando Amat
*
* \brief testing a full iteration of multi-view lucy richardson (without splitting image in blocks)
*
*/

#include <iostream>
#include <cstdint>
#include <time.h>       /* time_t, struct tm, difftime, time, mktime */
#include "multiviewDeconvolution.h"


using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "testing a full iteration of multi-view lucy richardson (without splitting image in blocks) running..." << std::endl;
	time_t start, end;

	//parameters
	string filepath("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/");
	int numIters = 2;
	int numViews = 4;
	float imgBackground = 100;
	cout << "===============TODO: activate total variations==============" << endl;
	float lambdaTV = -1.0;//0.008;
	string filePatternPSF( filepath + "psfReg_?.klb");
	string filePatternWeights(filepath + "weightsReg_?.klb");
	string filePatternImg(filepath + "imReg_?.klb");


	if (argc > 1)
		filepath = string(argv[1]);
	if (argc > 2)
		numIters = atoi(argv[2]);

	//declare object
	cout << "TODO: allow reading uint16 images and convert them to float on the fly" << endl;
	multiviewDeconvolution<float> *J;

	J = new multiviewDeconvolution<float>;

	//set number of views
	J->setNumberOfViews(numViews);

	//read images
	string filename;
	int err;
	for (int ii = 0; ii < numViews; ii++)
	{
		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternPSF, ii+1);
		err = J->readImage(filename, ii, std::string("psf"));//this function should just read image
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}

		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternWeights, ii+1);
		err = J->readImage(filename, ii, std::string("weight"));
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}

		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternImg, ii+1);
		err = J->readImage(filename, ii, std::string("img"));
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}
	}

	//upload everything to GPU and precompute as much as needed	
	cout << "Allocating workspace for deconvolution" << endl;
	time(&start);
	err = J->allocate_workspace(imgBackground);
	if (err > 0)
	{
		cout << "ERROR: allocating workspace" << endl;
		return err;
	}
	time(&end);
	cout << "Took " << difftime(end, start) << " secs" << endl;

	//compute deconvolution iterations
	cout << "Running multiviews deconvolution for " << numIters << " iterations" << endl;
	J->deconvolution_LR_TV(numIters, lambdaTV);
	
	//copy results from GPU to CPU 
	cout << " Copying deconvolution results from GPU to CPU" << endl;
	J->copyDeconvoutionResultToCPU();
	time(&end);
	cout << "Took " << difftime(end, start) << " secs" << endl;

	//save results	    
	cout << " Writing final result" << endl;
	err = J->writeDeconvoutionResult(string(filepath + "test_mv_deconv_LR.klb"));
	if (err > 0)
	{
		cout << "ERROR: writing result" << endl;
		return err;
	}

	//release memory
	delete J;


	return 0;
}