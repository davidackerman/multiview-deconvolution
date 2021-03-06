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
#include "commonCUDA.h"
#include "cuda.h"
#include "book.h"
#include "multiviewDeconvolution.h"


typedef float dataType;

using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "testing GPU convolution kernel in the GPU running..." << std::endl;

	int devCUDA = 0;
	
	//parameters
	string filepath("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/");
	if (argc > 1)
		filepath = string(argv[1]);
	

	string filePatternPSF(filepath + "psfReg_?.klb");
	string filePatternImg(filepath + "imReg_?.klb");
	int numViews = 1;

	//=====================================================================

	HANDLE_ERROR(cudaSetDevice(devCUDA));
	

	//declare object	
	multiviewDeconvolution<float> *J;

	J = new multiviewDeconvolution<float>;

	//set number of views
	J->setNumberOfViews(numViews);

	//read images
	string filename;
	int err;
	for (int ii = 0; ii < numViews; ii++)
	{
		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternPSF, ii + 1);
		err = J->readImage(filename, ii, std::string("psf"));//this function should just read image
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}

		filename = multiviewImage<float>::recoverFilenamePatternFromString(filePatternImg, ii + 1);
		err = J->readImage(filename, ii, std::string("img"));
		if (err > 0)
		{
			cout << "ERROR: reading file " << filename << endl;
			return err;
		}

		float* imConv = J->convolution3DfftCUDA_img_psf(ii, devCUDA);

		//write file
		char fileout[256];
		sprintf(fileout, "%sout_test_convPSFimg_view%d.raw", filepath.c_str(), ii+1);
		ofstream fid(fileout, ios::binary);
		fid.write((char*)imConv, J->numElements_img(ii) * sizeof(float));
		fid.close();

		cout << "Convolution results written successfully at " << fileout << endl;
		delete[] imConv;
	}

	delete J;

	//--------------------------------------------------
	//second test 
	std::cout << "testing GPU convolution kernel in the GPU running..." << std::endl;	
	filePatternPSF = string(filepath + "psfReg_1.klb");
	filePatternImg = string(filepath + "J_iter0000.klb");

	J = new multiviewDeconvolution<float>;

	//set number of views
	J->setNumberOfViews(1);

	//read images	
	err = J->readImage(filePatternPSF, 0, std::string("psf"));//this function should just read image
	if (err > 0)
	{
		cout << "ERROR: reading file " << filename << endl;
		return err;
	}	
	err = J->readImage(filePatternImg, 0, std::string("img"));
	if (err > 0)
	{
		cout << "ERROR: reading file " << filename << endl;
		return err;
	}

	float* imConv = J->convolution3DfftCUDA_img_psf(0, devCUDA);	
	//write file
	char fileout[256];
	sprintf(fileout, "%sout_test_convPSF_Jiter_.raw", filepath.c_str());
	ofstream fid(fileout, ios::binary);
	fid.write((char*)imConv, J->numElements_img(0) * sizeof(float));
	fid.close();

	cout << "Convolution results written successfully at " << fileout << endl;

	delete[] imConv;
	delete J;

	return 0;
}
