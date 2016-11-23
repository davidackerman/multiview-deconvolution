#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <string>
#include <math.h>
#include <stdbool.h>
#include "klb_Cwrapper.h"
#include "imwarp_flexible.h"

//typedef float imgType;

using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, const char** argv)
{
	// Just to satisfy my neurosis
	cout << "This is run_imwarp_flexible." << endl;

	// Process the arguments
	if (argc < 4)
	{
		cout << "ERROR: Need three arguments" << endl;
		return -1 ;
	}
	string inputFileName(argv[1]) ;
	string transformFileName(argv[2]) ;
	string outputFileName(argv[3]);

	// Define variable to read images
	uint32_t xyzct[KLB_DATA_DIMS];
	KLB_DATA_TYPE dataType;
	float32_t pixelSize[KLB_DATA_DIMS];
	uint32_t blockSize[KLB_DATA_DIMS];
	KLB_COMPRESSION_TYPE compressionType;
	char metadata[KLB_METADATA_SIZE];

	// Read the input stack
	cout << "Reading input file..." << endl;
	float* input_stack = (float*)readKLBstack(inputFileName.c_str(), xyzct, &dataType, -1, pixelSize, blockSize, &compressionType, metadata);
	if (dataType != KLB_DATA_TYPE::FLOAT32_TYPE || input_stack == NULL)
	{
		cout << "ERROR: input image is not single-precision float" << endl;
		return 2;
	}
	
	// Open the file that holds the the affine transform matrix (which should be row form, i.e. the last col should be [0 0 0 1]')
	// In the file, A is in row-major order, one row per line, but imwarp_flexible() wants A in col-major order
	// So we take this into account when we read A in.
	cout << "Loading affine transformation..." << endl;
	float A[AFFINE_3D_MATRIX_SIZE];
	ifstream fin(transformFileName.c_str());
	if (!fin.is_open())
	{
		cout << "ERROR: opening file with affine transform " << transformFileName << endl;
		return 3;
	}
	for (int ii = 0; ii < 4; ii++)  // row index
		for (int jj = 0; jj < 4; jj++)  // col index
			fin >> A[4*jj + ii] ;  // appropriate form for col-major indexing
	fin.close();
	print_affine_3d_matrix(A) ;

    // Set the input, output dims
    int64_t input_dims[3] = { xyzct[0], xyzct[1], xyzct[2] };  // TODO: Is this right?  Do we need to swap x and y dims?
    int64_t output_dims[3] = { xyzct[0], xyzct[1], xyzct[2] };  // same as input_dims
    
    // Allocate memory for output stack
	cout << "Applying affine transformation..." << endl;
	size_t n_output_elements = (size_t) output_dims[0] * output_dims[1] * output_dims[2] ;
	float* output_stack = new float[n_output_elements] ;

	// Call imwarp in CPU
	float input_origin[3] = { 0.5, 0.5, 0.5 };
	float input_spacing[3] = { 1.0, 1.0, 1.0 };
	float output_origin[3] = { 0.5, 0.5, 0.5 };
	float output_spacing[3] = { 1.0, 1.0, 1.0 };
	//int interpolation_mode = 2;  // cubic interpolation, no black background
	bool is_cubic = true ;
	bool is_background_black = false ;
	auto t1 = Clock::now();
	imwarp_flexible(input_stack, input_dims, input_origin, input_spacing,
		            output_stack, output_dims, output_origin, output_spacing,
					A, 
					is_cubic, is_background_black) ;
	auto t2 = Clock::now();
	std::cout << "Imwarp fast in CPU with "<<getNumberOfCores()<< " threads  took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"<< std::endl;

	//write out solution
	cout << "Writing out solution..." << endl;
	//string filenameOut(outputFileName); 
	writeKLBstack(output_stack, outputFileName.c_str(), xyzct, KLB_DATA_TYPE::FLOAT32_TYPE, -1, NULL, NULL, KLB_COMPRESSION_TYPE::BZIP2, NULL);

	// Release memory
	delete[] output_stack;
	free(input_stack);

	std::cout << "...OK" << endl;
	return 0;
}
