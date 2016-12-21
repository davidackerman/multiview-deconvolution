#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <string>
#include <memory>
#include <math.h>
#include <stdbool.h>
#include "klb_Cwrapper.h"
#include "external/imwarp_flexible/imwarp_flexible.h"

//typedef float imgType;

using namespace std;
typedef std::chrono::high_resolution_clock Clock;

template<typename T>
unique_ptr<T> read_col_major_matrix(const T* dummy, int64_t n_rows, int64_t n_cols, const string & file_name)
{
    ifstream input_file(file_name.c_str(), ios::binary);
    if (!input_file.is_open())
    {
        cerr << "ERROR: opening file " << file_name << endl;
        throw std::runtime_error("Error opening file") ;
    }
    int64_t n_els = n_rows*n_cols ;
    unique_ptr<T> matrix(new T[n_els]) ;
    T* raw_matrix(matrix.get()) ;
    input_file.read((char*)(raw_matrix), n_els*sizeof(T)) ;
    input_file.close();

    return matrix ;
}

template unique_ptr<double> read_col_major_matrix(const double* dummy, int64_t n_rows, int64_t n_cols, const string & file_name) ;
template unique_ptr<float> read_col_major_matrix(const float* dummy, int64_t n_rows, int64_t n_cols, const string & file_name) ;
template unique_ptr<int64_t> read_col_major_matrix(const int64_t* dummy, int64_t n_rows, int64_t n_cols, const string & file_name) ;

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
	cout << "This is run_imwarp_flexible." << endl;

	// Process the arguments
	if (argc < 9)
	{
		cout << "ERROR: Need eight arguments" << endl;
		return -1 ;
	}
	string input_stack_file_name(argv[1]) ;
    string input_origin_file_name(argv[2]) ;
    string input_spacing_file_name(argv[3]) ;
	string transform_file_name(argv[4]) ;
	string output_origin_file_name(argv[5]) ;
    string output_spacing_file_name(argv[6]) ;
    string output_dimensions_file_name(argv[7]) ;
	string output_stack_file_name(argv[8]);

	// Read the input stack
    int64_t input_dims[3] ;
    float* input_stack ;
    int err = read_3d_float_klb_stack(input_dims, &input_stack, input_stack_file_name) ;
    if (err)
    {
        cout << "ERROR: There was a problem reading the input stack" << endl;
        return -1;
    }

    // Read the input origin
    unique_ptr<double> input_origin(read_col_major_matrix((double*)(0), 1, 3, input_origin_file_name)) ;

    // Read the input spacing
    unique_ptr<double> input_spacing(read_col_major_matrix((double*)(0), 1, 3, input_spacing_file_name)) ;

    // Read the transform matrix (which should be row form, i.e. the last col should be [0 0 0 1]', 
    // and the translation should be in the last row).  We assume this is stored col-major, which is what
    // imwarp_flexible() wants.
    unique_ptr<double> T_in_row_form(read_col_major_matrix((double*)(0), 4, 4, transform_file_name)) ;

    // Read the output origin
    unique_ptr<double> output_origin(read_col_major_matrix((double*)(0), 1, 3, output_origin_file_name)) ;

    // Read the output spacing
    unique_ptr<double> output_spacing(read_col_major_matrix((double*)(0), 1, 3, output_spacing_file_name)) ;

    // Read the output dims
	// Open the file that holds the output dimensions, which should be in the order n_y, n_x, n_z
	cout << "Loading output dimensions..." << endl;
    unique_ptr<int64_t> output_dims(read_col_major_matrix((int64_t*)(0), 1, 3, output_dimensions_file_name)) ;
    int64_t* raw_output_dims = output_dims.get() ;

	// Check that the output dims are small enough to fit in a KLB file
    if (raw_output_dims[0]>UINT32_MAX || raw_output_dims[1]>UINT32_MAX || raw_output_dims[2]>UINT32_MAX)
	{
		cout << "ERROR: requested output dimensions are too large for KLB file" << endl;
		return 5 ;
	}

    // Allocate memory for output stack
	cout << "Applying affine transformation..." << endl;
    size_t n_output_elements = (size_t)(raw_output_dims[0] * raw_output_dims[1] * raw_output_dims[2]) ;
	float* output_stack = new float[n_output_elements] ;

	// Call imwarp_flexible(), which does everything on the CPU
	//int interpolation_mode = 2;  // cubic interpolation, no black background
	bool is_cubic = true ;
	bool is_background_black = true ;
	auto t1 = Clock::now();
    imwarp_flexible(
        input_stack, input_dims, input_origin.get(), input_spacing.get(),
        output_stack, output_dims.get(), output_origin.get(), output_spacing.get(),
        T_in_row_form.get(),
		is_cubic, is_background_black) ;
	auto t2 = Clock::now() ;
    std::cout << "Imwarp fast in CPU with " << get_number_of_cores() << " threads  took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl ;

	// Write out solution
	cout << "Writing out solution..." << endl;
	//string filenameOut(outputFileName); 
    uint32_t output_yxzct[KLB_DATA_DIMS] = { (uint32_t)raw_output_dims[0], (uint32_t)raw_output_dims[1], (uint32_t)raw_output_dims[2], 1, 1 } ;
	writeKLBstack(output_stack, output_stack_file_name.c_str(), output_yxzct, KLB_DATA_TYPE::FLOAT32_TYPE, -1, NULL, NULL, KLB_COMPRESSION_TYPE::BZIP2, NULL);

	// Release memory
	free(input_stack);
    //free(input_origin);
    //free(input_spacing);
    //free(T_in_row_form);
    //free(output_origin);
    //free(output_spacing);

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
