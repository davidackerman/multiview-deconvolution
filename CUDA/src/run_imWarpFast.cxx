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
#include <math.h>
#include "klb_Cwrapper.h"
#include "affine_transform_3d_single.h"

typedef float imgType;



using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, const char** argv)
{
	std::cout << "test imwarpfast running running..." << std::endl;

	//parameters	
	//string imPath("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/imwarp/");
    //string imPath("D:/Adam/mvd/CUDA/test/imwarp-cpp-test-data/");
    //string imInFilename("imOrig.klb");
    //string imInFilename("input.klb");
    //string outputFileNameMatlabFilename("imTransformed_matlab.klb");
    //string outputFileNameMatlabFilename("target_psf.klb");
    //int64_t dimsOut[3] = { 704, 1504, 585 };
    //int64_t dimsOut[3] = { 160, 41, 37 } ;
    //string transformFileName("imOrig_Affine.txt");
    //string transformFileName("affine_transform_matrix_in_y_equals_Ax_form_row_major.txt");
    //string outputFileName("imTransformed_C.klb");
    //string outputFileName("output.klb");
    int interpMode = 2;


	if (argc < 4)
	{
		cout << "ERROR: Need three arguments" << endl;
		return -1 ;
	}
	//imPath = string(argv[1]) ;
	string inputFileName(argv[1]) ;
	string transformFileName(argv[2]) ;
	string outputFileName(argv[3]);

	//define variable to read images
	uint32_t xyzct[KLB_DATA_DIMS];
	//uint32_t xyzctOut[KLB_DATA_DIMS];
	KLB_DATA_TYPE dataType;
	float32_t pixelSize[KLB_DATA_DIMS];
	uint32_t blockSize[KLB_DATA_DIMS];
	KLB_COMPRESSION_TYPE compressionType;
	char metadata[KLB_METADATA_SIZE];

	cout << "Reading input file..." << endl;
	//string filename(inputFileName);	
	float* imIn = (float*)readKLBstack(inputFileName.c_str(), xyzct, &dataType, -1, pixelSize, blockSize, &compressionType, metadata);
	if (dataType != KLB_DATA_TYPE::FLOAT32_TYPE || imIn == NULL)
	{
		cout << "ERROR: input image is not single-precision float" << endl;
		return 2;
	}
	
	/*
	cout << "Converting input image to float" << endl;
	size_t imSize = 1;
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		imSize *= (uint64_t)(xyzct[ii]);
	float* imInFloat = new float[imSize];
	for (size_t ii = 0; ii < imSize; ii++)
		imInFloat[ii] = (float)(imIn[ii]);
	free(imIn);
	*/

	cout << "Loading affine transformation..." << endl;
	float A[AFFINE_3D_MATRIX_SIZE];
	//filename = string(transformFileName);
	ifstream fin(transformFileName.c_str());
	if (!fin.is_open())
	{
		cout << "ERROR: opening file with affine transform " << transformFileName << endl;
		return 3;
	}

	// In the file, A is in row-major order, one row per line, but this code seems to want A in col-major order
	// (That's how affine3d_printMatrix() prints things.)  So we take this into account when we read A in.
	for (int ii = 0; ii < 4; ii++)
		for (int jj = 0; jj < 4; jj++)
			fin >> A[4*jj + ii];
	fin.close();
	affine3d_printMatrix(A);

    // Set the input, output dims
    int64_t dimsIn[3] = { xyzct[0], xyzct[1], xyzct[2] };
    int64_t dimsOut[3] = { xyzct[0], xyzct[1], xyzct[2] };  // same as dimsIn
    
    //allocate memory
	cout << "Applying affine transformation..." << endl;
	size_t imOutSize = 1;
	for (int ii = 0; ii < 3; ii++)
		imOutSize *= dimsOut[ii];
	float* imOut = new float[imOutSize];

	//call imwarp in CPU

	auto t1 = Clock::now();
	imwarpFast_MatlabEquivalent(imIn, imOut, dimsIn, dimsOut, A, interpMode);
	auto t2 = Clock::now();

	std::cout << "Imwarp fast in CPU with "<<getNumberOfCores()<< " threads  took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"<< std::endl;

	
	//cout << "Reading output file from Matlab for verification..." << endl;
	//filename = string(imPath + outputFileNameMatlabFilename);
	//float* imOutMatlab = (float *)readKLBstack(filename.c_str(), xyzctOut, &dataType, -1, pixelSize, blockSize, &compressionType, metadata);
	//if (dataType != KLB_DATA_TYPE::FLOAT32_TYPE || imOutMatlab == NULL)
	//{
	//	cout << "ERROR: output image is not float" << endl;
	//	return 2;
	//}

	//write out solution
	cout << "Writing out solution..." << endl;
	//string filenameOut(outputFileName); 
	writeKLBstack(imOut, outputFileName.c_str(), xyzct, KLB_DATA_TYPE::FLOAT32_TYPE, -1, NULL, NULL, KLB_COMPRESSION_TYPE::BZIP2, NULL);


	//cout << "Comparing solutions..." << endl;
	//for (size_t ii = 0; ii < imOutSize; ii++)
	//{
	//	if (fabs(imOut[ii] - imOutMatlab[ii]) > 1e-1)
	//	{
	//		cout << "ERROR: position " << ii << " solutions disagree with I_matlab = " << imOutMatlab[ii] << " and I_C = " << imOut[ii] << " and diff = " << fabs(imOut[ii] - imOutMatlab[ii]) << endl;
	//		return 4;
	//	}
	//}

	//release memory
	//delete[] imInFloat;
	delete[] imOut;
	//free(imOutMatlab);
	free(imIn);


	std::cout << "...OK" << endl;
	return 0;
}
