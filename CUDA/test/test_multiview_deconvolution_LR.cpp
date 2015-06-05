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
#include "multiviewDeconvolution.h"


using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "testing a full iteration of multi-view lucy richardson (without splitting image in blocks) running..." << std::endl;

	//parameters
	int numIters = 1;
	int numViews = 4;
	float lambdaTV = 0.008;
	string filePatternPSF = "write something";
	string filePatternWeights = "write something";
	string filePatternImg = "write something";

	//declare object
	cout << "TODO: use a derived class implementing LR" << endl;
	multiviewDeconvolution<uint16_t> *J;

	J = new multiviewDeconvolution<uint16_t>;

	//set number of views
	J->setNumberOfViews(numViews);

	//read images
	string filename;
	int err;
	for (int ii = 0; ii < numViews; ii++)
	{
		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternPSF, ii);
		err = J->readImage(filename, ii, std::string("psf"));//this function should just read image
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}

		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternWeights, ii);
		err = J->readImage(filename, ii, std::string("weight"));
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}

		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternImg, ii);
		err = J->readImage(filename, ii, std::string("img"));
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}
	}

	//upload everything to GPU and precompute as much as needed
	err = J->allocate_workspace();
	if (err > 0)
	{
		cout << "ERROR: allocating workspace" << endl;
		return err;
	}

	//compute deconvolution iterations
	J->deconvolution_LR_TV(numIters, lambdaTV);

	//release memory
	delete J;


	return 0;
}