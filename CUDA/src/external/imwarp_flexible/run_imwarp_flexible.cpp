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
	if (argc < 7)
	{
		cout << "ERROR: Need six arguments" << endl;
		return -1 ;
	}
	string inputFileName(argv[1]) ;
	string inputOriginAndSpacingFileName(argv[2]) ;
	string transformFileName(argv[3]) ;
	string outputOriginAndSpacingFileName(argv[4]) ;
	string outputDimensionsFileName(argv[5]) ;
	string outputFileName(argv[6]);

	// Read the input stack
	cout << "Reading input file..." << endl;
	uint32_t input_yxzct[KLB_DATA_DIMS];
	KLB_DATA_TYPE input_data_type;
	float32_t input_pixel_size[KLB_DATA_DIMS];
	uint32_t input_block_size[KLB_DATA_DIMS];
	KLB_COMPRESSION_TYPE input_compression_type;
	char input_metadata[KLB_METADATA_SIZE];
	float* input_stack = (float*)readKLBstack(inputFileName.c_str(), input_yxzct, &input_data_type, -1, input_pixel_size, input_block_size, &input_compression_type, input_metadata);
	if (input_data_type != KLB_DATA_TYPE::FLOAT32_TYPE || input_stack == NULL)
	{
		cout << "ERROR: input image is not single-precision float" << endl;
		return 2;
	}

	// Get the input dims in the form we need
	int64_t input_dims[3] = { input_yxzct[0], input_yxzct[1], input_yxzct[2] };
	cout << "Input stack dimensions (input_dims):     y:" << input_dims[0] << "    x:" << input_dims[1] << "    z:" << input_dims[2] << endl ;

	// Open the file that holds the input origin and spacing
	// The text file should be a 3x2 matrix in row-major order, one row per line, with the origin being the first col, and spacing being the second.
	// Each vector should be in x, y, z order.
	cout << "Loading input origin and spacing..." << endl;
	float input_origin[3] = { 0.5, 0.5, 0.5 };
	float input_spacing[3] = { 1.0, 1.0, 1.0 };
	ifstream f_in_aux(inputOriginAndSpacingFileName.c_str());
	if (!f_in_aux.is_open())
	{
		cout << "ERROR: opening file with input origin and spacing " << inputOriginAndSpacingFileName << endl;
		return 4;
	}
	for (int i = 0; i < 3; i++)  // row index
	{
		f_in_aux >> input_origin[i] ;
		f_in_aux >> input_spacing[i] ;
	}
	f_in_aux.close();

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
			fin >> A[4 * jj + ii] ;  // appropriate form for col-major indexing
	fin.close();
	print_affine_3d_matrix(A) ;

	// Open the file that holds the output origin and spacing
    // The text file should be a 3x2 matrix in row-major order, one row per line, with the origin being the first col, and spacing being the second.
	// Each vector should be in x, y, z order.
	cout << "Loading output origin and spacing..." << endl;
	float output_origin[3] = { 0.5, 0.5, 0.5 };
	float output_spacing[3] = { 1.0, 1.0, 1.0 };
	ifstream f_out_aux(outputOriginAndSpacingFileName.c_str());
	if (!f_out_aux.is_open())
	{
		cout << "ERROR: opening file with output origin and spacing " << outputOriginAndSpacingFileName << endl;
		return 4;
	}
	for (int i = 0; i < 3; i++)  // row index
	{
		f_out_aux >> output_origin[i] ;
		f_out_aux >> output_spacing[i] ;
	}
	f_out_aux.close();

	// Open the file that holds the output dimensions, which should be in the order n_y, n_x, n_z
	cout << "Loading output dimensions..." << endl;
	int64_t output_dims[3] ;
	ifstream f_out_dims(outputDimensionsFileName.c_str());
	if (!f_out_dims.is_open())
	{
		cout << "ERROR: opening file with output dimensions " << outputDimensionsFileName << endl;
		return 4;
	}
	for (int i = 0; i < 3; i++)  // row index
	{
		f_out_dims >> output_dims[i] ;
	}
	f_out_dims.close();
    
	// Check that the output dims are small enough to fit in a KLB file
	if ( output_dims[0]>UINT32_MAX || output_dims[1]>UINT32_MAX || output_dims[2]>UINT32_MAX )
	{
		cout << "ERROR: requested output dimensions are too large for KLB file" << endl;
		return 5 ;
	}

    // Allocate memory for output stack
	cout << "Applying affine transformation..." << endl;
	size_t n_output_elements = (size_t) output_dims[0] * output_dims[1] * output_dims[2] ;
	float* output_stack = new float[n_output_elements] ;

	// Call imwarp_flexible(), which does everything on the CPU
	//int interpolation_mode = 2;  // cubic interpolation, no black background
	bool is_cubic = true ;
	bool is_background_black = true ;
	auto t1 = Clock::now();
	imwarp_flexible(input_stack, input_dims, input_origin, input_spacing,
		            output_stack, output_dims, output_origin, output_spacing,
					A, 
					is_cubic, is_background_black) ;
	auto t2 = Clock::now() ;
	std::cout << "Imwarp fast in CPU with " << getNumberOfCores() << " threads  took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"<< std::endl ;

	// Write out solution
	cout << "Writing out solution..." << endl;
	//string filenameOut(outputFileName); 
	uint32_t output_yxzct[KLB_DATA_DIMS] = { (uint32_t)output_dims[0], (uint32_t)output_dims[1], (uint32_t)output_dims[2], 1, 1 } ;
	writeKLBstack(output_stack, outputFileName.c_str(), output_yxzct, KLB_DATA_TYPE::FLOAT32_TYPE, -1, NULL, NULL, KLB_COMPRESSION_TYPE::BZIP2, NULL);

	// Release memory
	free(input_stack);

	// Read the output stack back in, to make sure we can
	cout << "Reading output file back in, as sanity check..." ;
	uint32_t output_check_yxzct[KLB_DATA_DIMS];
	KLB_DATA_TYPE output_check_data_type;
	float32_t output_check_pixel_size[KLB_DATA_DIMS];
	uint32_t output_check_block_size[KLB_DATA_DIMS];
	KLB_COMPRESSION_TYPE output_check_compression_type;
	char output_check_metadata[KLB_METADATA_SIZE];
	float* output_check_stack = (float*)readKLBstack(outputFileName.c_str(), output_check_yxzct, &output_check_data_type, -1, output_check_pixel_size, output_check_block_size, &output_check_compression_type, output_check_metadata);
	if (output_check_data_type != KLB_DATA_TYPE::FLOAT32_TYPE || output_check_stack == NULL)
	{
		cout << "ERROR: output stack, as read back from disk, is not single-precision float" << endl;
		return 6 ;
	}
	for (size_t i = 0; i < KLB_DATA_DIMS; ++i)
	{	
		if ( output_check_yxzct[i] != output_yxzct[i] )
		{
			cout << "ERROR: Dimensions of output stack, as read back from disk, do not match dimensions of original output stack" << endl;
			return 7 ;
		}
	}
	// If the dimensions match, can use the already-computed number of output elements
	for (size_t i = 0; i < n_output_elements; ++i)
	{
		if (output_check_stack[i] != output_stack[i])
		{
			cout << "ERROR: Data in output stack, as read back from disk, does not match original output stack at element" << i << "(using zero-based indexing)" << endl ;
			return 8 ;
		}
	}
	cout << "done, and matches original output stack." << endl ;

	// Release memory allocated by us in main() via new operator, and memory allocated in readKLBstack() via malloc()
	delete[] output_stack;
	free(output_check_stack) ;

	// Output final message, and execute a normal return
	std::cout << "...OK" << endl;
	return 0;
}
