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
#include <chrono>
#include "multiviewImage.h"
#include "multiviewDeconvolution.h"

typedef float imgType;



using namespace std;
typedef std::chrono::high_resolution_clock Clock;

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
	auto t1 = Clock::now();
	outputType* TV_CPU = multiviewDeconvolution<imgType>::debug_regularization_TV_CPU(img.getPointer_CPU(0), img.dimsImgVec[0].dims);
	auto t2 = Clock::now();

	std::cout << "TV regularization with finite differences in CPU took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"<< std::endl;

	//write out solution
	string filenameOut(filepath + "test_TV_output_CPU_finiteDiff.raw"); 
	multiviewDeconvolution<imgType>::debug_writeCPUarray(TV_CPU, img.dimsImgVec[0], filenameOut);

	//release memory
	delete[] TV_CPU;


	//test GPU code
	t1 = Clock::now();
	multiviewDeconvolution<float> TV_GPU;
	TV_GPU.debug_regularization_TV_GPU(img.getPointer_CPU(0), img.dimsImgVec[0].dims);
	t2 = Clock::now();
	std::cout << "TV regularization with Gaussian derivatives in GPU took (including I/O) " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

	//write out solution
	filenameOut = string(filepath + "test_TV_output_GPU.raw");
	TV_GPU.debug_writDeconvolutionResultRaw(filenameOut);	


	std::cout << "...OK" << endl;
	return 0;
}