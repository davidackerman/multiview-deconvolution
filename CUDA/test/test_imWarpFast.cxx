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
#include <fstream>
#include <cstdint>
#include <chrono>
#include <string>
#include "klb_Cwrapper.h"
#include "affine_transform_3d_single.h"

typedef float imgType;



using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, const char** argv)
{
	std::cout << "test imwarpfast running running..." << std::endl;

	//parameters	
	string imPath("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/imwarp/");
	string imInFilename("imOrig.klb");
	string imResultMatlabFilename("imTransformed_matlab.klb");
	int64_t dimsOut[3] = { 704    ,    1504 ,        585 };
	string affineTxt("imOrig_Affine.txt");
	string imResult("imTransformed_C.klb");
	int interpMode = 2;


	if (argc > 1)
		imPath = string(argv[1]);

	//define variable to read images
	uint32_t xyzct[KLB_DATA_DIMS];
	uint32_t xyzctOut[KLB_DATA_DIMS];
	KLB_DATA_TYPE dataType;
	float32_t pixelSize[KLB_DATA_DIMS];
	uint32_t blockSize[KLB_DATA_DIMS];
	KLB_COMPRESSION_TYPE compressionType;
	char metadata[KLB_METADATA_SIZE];

	cout << "Reading input file..." << endl;
	string filename(imPath + imInFilename);	
	uint16_t* imIn = (uint16_t*)readKLBstack(filename.c_str(), xyzct, &dataType, -1, pixelSize, blockSize, &compressionType, metadata);
	if (dataType != KLB_DATA_TYPE::UINT16_TYPE || imIn == NULL)
	{
		cout << "ERROR: input image is not uint16" << endl;
		return 2;
	}
	
	cout << "Converting input image to float" << endl;
	size_t imSize = 1;
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		imSize *= (uint64_t)(xyzct[ii]);
	float* imInFloat = new float[imSize];
	for (size_t ii = 0; ii < imSize; ii++)
		imInFloat[ii] = (float)(imIn[ii]);

	free(imIn);


	cout << "Loading affine transformation..." << endl;
	float A[AFFINE_3D_MATRIX_SIZE];
	filename = string(imPath + affineTxt);
	ifstream fin(filename.c_str()); 
	if (!fin.is_open())
	{
		cout << "ERROR: opening file with affine transform " << filename << endl;
		return 3;
	}

	for (int ii = 0; ii < AFFINE_3D_MATRIX_SIZE; ii++)
		fin >> A[ii];
	fin.close();
	affine3d_printMatrix(A);

	//allocate memory
	cout << "Applying affine transformation..." << endl;
	size_t imOutSize = 1;
	for (int ii = 0; ii < 3; ii++)
		imOutSize *= dimsOut[ii];
	float* imOut = new float[imOutSize];

	//call imwarp in CPU
	//int64_t dimsOut[3] = { xyzctOut[0], xyzctOut[1], xyzctOut[2]};
	int64_t dimsIn[3] = { xyzct[0], xyzct[1], xyzct[2] };

	auto t1 = Clock::now();
	imwarpFast_MatlabEquivalent(imInFloat, imOut, dimsIn, dimsOut, A, interpMode);
	auto t2 = Clock::now();

	std::cout << "Imwarp fast in CPU with "<<getNumberOfCores()<< " threads  took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"<< std::endl;

	
	cout << "Reading output file from Matlab for verification..." << endl;
	filename = string(imPath + imResultMatlabFilename);
	float* imOutMatlab = (float *)readKLBstack(filename.c_str(), xyzctOut, &dataType, -1, pixelSize, blockSize, &compressionType, metadata);
	if (dataType != KLB_DATA_TYPE::FLOAT32_TYPE || imOutMatlab == NULL)
	{
		cout << "ERROR: output image is not float" << endl;
		return 2;
	}

	//write out solution
	cout << "Writing out solution..." << endl;
	string filenameOut(imPath  + imResult); 
	writeKLBstack(imOut, filenameOut.c_str(), xyzctOut, KLB_DATA_TYPE::FLOAT32_TYPE, -1, NULL, NULL, KLB_COMPRESSION_TYPE::BZIP2, NULL);


	cout << "Comparing solutions..." << endl;
	for (size_t ii = 0; ii < imOutSize; ii++)
	{
		if (fabs(imOut[ii] - imOutMatlab[ii]) > 1e-1)
		{
			cout << "ERROR: position " << ii << " solutions disagree with I_matlab = " << imOutMatlab[ii] << " and I_C = " << imOut[ii] << " and diff = " << fabs(imOut[ii] - imOutMatlab[ii]) << endl;
			return 4;
		}
	}

	//release memory
	delete[] imInFloat;
	delete[] imOut;
	free(imOutMatlab);


	std::cout << "...OK" << endl;
	return 0;
}