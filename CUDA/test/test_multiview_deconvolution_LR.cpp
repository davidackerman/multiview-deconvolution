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
	string filePatternPSF = "write something";
	string filePatternWeights = "write something";
	string filePatternImg = "write something";

	//declare object
	cout << "TODO: use a derived class implementing LR" << endl;
	multiviewDeconvolution<uint16_t> *J;

	J = new multiviewDeconvolution<uint16_t>;

	//set number of views
	J.setNumberOfViews(numViews);

	//read images
	string filename;
	for (int ii = 0; ii < numViews; ii++)
	{
		filename = recoverFilenamePatternFromString(filePatternPSF, ii);
		J.readImage(filename, ii, 'psf');//this function should just read image

		filename = recoverFilenamePatternFromString(filePatternWeights, ii);
		J.readImage(filename, ii, 'weight');

		filename = recoverFilenamePatternFromString(filePatternImg, ii);
		J.readImage(filename, ii, 'img');
	}

	//upload everything to GPU and precompute as much as needed
	J.allocate_workspace();

	//compute deconvolution iterations
	J.deconvolution(numIters);

	//release memory
	delete J;
}