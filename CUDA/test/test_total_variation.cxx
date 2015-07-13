/*
*
* Authors: Fernando Amat
*  test_load_multiview_images.cpp
*
*  Created on : June 5th, 2015
* Author : Fernando Amat
*
* \brief calculate total variation and compare it to C code
*
*/

#include <iostream>
#include <cstdint>
#include "multiviewImage.h"
#include "multiviewDeconvolution.h"

typedef float imgType;

using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "test Total variation running running..." << std::endl;

	//parameters
	int numViews = 1;
	string filepath("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/");

	if (argc > 1)
		filepath = string(argv[1]);

	string filePattern( filepath + "imReg_?.klb" );

	//declare object
	multiviewImage<imgType> img;

	//read object
	for (int ii = 0; ii < numViews; ii++)
	{
		string filename = img.recoverFilenamePatternFromString(filePattern, ii+1);
		int err = img.readImage(filename, -1);

		if (err != 0)
		{
			cout << "ERROR: loading image " << filename << endl;
			return err;
		}
	}

	//call totalvariation in CPU
	outputType* TV_CPU = multiviewDeconvolution<imgType>::debug_regularization_TV_CPU(img.getPointer_CPU(0), img.dimsImgVec[0].dims);

	//write out solution
	string filenameOut(filepath + "test_TV_output.raw"); 
	multiviewDeconvolution<imgType>::debug_writeCPUarray(TV_CPU, img.dimsImgVec[0], filenameOut);

	//release memory
	delete[] TV_CPU;

	std::cout << "...OK" << endl;
	return 0;
}