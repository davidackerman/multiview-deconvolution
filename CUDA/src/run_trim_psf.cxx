#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <string>
#include <math.h>
#include <stdbool.h>
#include "klb_Cwrapper.h"
#include "imgUtils.h"

//typedef float imgType;

using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int read_3d_float_klb_stack(int64_t* dims, float** stack_ptr, const string & file_name)
{
    // dims needs to pre-allocated to length 3.
    // stack_ptr should point to a pre-allocated float*.
    // On return, *stack_ptr will be a float* to memory that has been allocated with malloc(), which will need to be free()'d by the caller at some point.
    // The return value is 0 if all went well, and a nonzero error code if not.

    // Read the input stack
    cout << "Reading input file..." << endl;
    uint32_t input_yxzct[KLB_DATA_DIMS];
    KLB_DATA_TYPE input_data_type;
    float32_t input_pixel_size[KLB_DATA_DIMS];
    uint32_t input_block_size[KLB_DATA_DIMS];
    KLB_COMPRESSION_TYPE input_compression_type;
    char input_metadata[KLB_METADATA_SIZE];
    float* input_stack = (float*)readKLBstack(file_name.c_str(), input_yxzct, &input_data_type, -1, input_pixel_size, input_block_size, &input_compression_type, input_metadata);
    if (input_data_type != KLB_DATA_TYPE::FLOAT32_TYPE || input_stack == NULL)
    {
        cout << "ERROR: input image is not single-precision float" << endl;
        return 2;
    }
    *stack_ptr = input_stack ;

    // Get the input dims in the form we need
    dims[0] = input_yxzct[0] ;
    dims[1] = input_yxzct[1] ;
    dims[2] = input_yxzct[2] ;
    cout << "Input stack dimensions (dims):     y:" << dims[0] << "    x:" << dims[1] << "    z:" << dims[2] << endl ;

    return 0 ;
}

int main(int argc, const char** argv)
{
	// Just to satisfy my neurosis
	cout << "This is run_trim_psf." << endl;

	// Process the arguments
	if (argc < 3)
	{
		cout << "ERROR: Need two arguments" << endl;
		return -1 ;
	}
	string input_stack_file_name(argv[1]) ;
	string output_stack_file_name(argv[2]);

	// Read the input stack
    int64_t input_dims[3] ;
    float* input_stack ;
    int err = read_3d_float_klb_stack(input_dims, &input_stack, input_stack_file_name) ;
    if (err)
    {
        cout << "ERROR: There was a problem reading the input stack" << endl;
        return -1;
    }

    // Do the trimming
    int64_t output_dims[3] ;
    float *output_stack = trim_psf_3d(output_dims, input_stack, input_dims) ;

	// Write out the output stack
	cout << "Writing output stack..." << endl;
	uint32_t output_yxzct[KLB_DATA_DIMS] = { (uint32_t)output_dims[0], (uint32_t)output_dims[1], (uint32_t)output_dims[2], 1, 1 } ;
	writeKLBstack(output_stack, output_stack_file_name.c_str(), output_yxzct, KLB_DATA_TYPE::FLOAT32_TYPE, -1, NULL, NULL, KLB_COMPRESSION_TYPE::BZIP2, NULL);

	// Release memory for input stack
	free(input_stack);

	// Read the output stack back in, to make sure we can
	cout << "Reading output file back in, as sanity check..." ;
	uint32_t output_check_yxzct[KLB_DATA_DIMS];
	KLB_DATA_TYPE output_check_data_type;
	float32_t output_check_pixel_size[KLB_DATA_DIMS];
	uint32_t output_check_block_size[KLB_DATA_DIMS];
	KLB_COMPRESSION_TYPE output_check_compression_type;
	char output_check_metadata[KLB_METADATA_SIZE];
	float* output_check_stack = (float*)readKLBstack(output_stack_file_name.c_str(), output_check_yxzct, &output_check_data_type, -1, output_check_pixel_size, output_check_block_size, &output_check_compression_type, output_check_metadata);
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
	// If the dimensions match, check elements
    int64_t n_output_elements = element_count_from_dims_3d(output_dims) ;
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
