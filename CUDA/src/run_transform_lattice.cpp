#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <string>
#include <memory>
#include <math.h>
#include <stdbool.h>
#include "klb_Cwrapper.h"
#include "imgUtils.h"

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
    //int64_t n = 0 ;
    //for (int64_t j = 0; j < n_cols; ++j)  // col index
    //{
    //    for (int64_t i = 0; i < n_rows; ++i)  // row index
    //    {
    //        //input >> raw_matrix[n] ;
    //        input.read(raw_matrix, n_els*sizeof(T)) ;
    //        ++n ;
    //    }
    //}
    input_file.close();

    return matrix ;
}

template<typename T>
void write_col_major_matrix(const string & file_name, const unique_ptr<T> & matrix, int64_t n_rows, int64_t n_cols)
{
    ofstream output_file(file_name.c_str(), ios::binary);
    if (!output_file.is_open())
    {
        cerr << "ERROR: opening file " << file_name << " for writing" << endl;
        throw std::runtime_error("Error opening file for writing") ;
    }
    T* raw_matrix(matrix.get()) ;
    int64_t n_els = n_rows*n_cols ;
    output_file.write((char*)(raw_matrix), n_els*sizeof(T)) ;
    //int64_t n = 0 ;
    //for (int64_t j = 0; j < n_cols; ++j)  // col index
    //{
    //    for (int64_t i = 0; i < n_rows; ++i)  // row index
    //    {
    //        output_file << raw_matrix[n] ;
    //        ++n ;
    //    }
    //}
    output_file.close();
}

template unique_ptr<double> read_col_major_matrix(const double* dummy, int64_t n_rows, int64_t n_cols, const string & file_name) ;
template unique_ptr<float> read_col_major_matrix(const float* dummy, int64_t n_rows, int64_t n_cols, const string & file_name) ;
template unique_ptr<int64_t> read_col_major_matrix(const int64_t* dummy, int64_t n_rows, int64_t n_cols, const string & file_name) ;

template void write_col_major_matrix(const string & file_name, const unique_ptr<int64_t> & matrix, int64_t n_rows, int64_t n_cols) ;
template void write_col_major_matrix(const string & file_name, const unique_ptr<double> & matrix, int64_t n_rows, int64_t n_cols) ;

int main(int argc, const char** argv)
{
	// Just to satisfy my neurosis
	cout << "This is run_transform_lattice." << endl;

    // Typical invocation: run_transform_lattice T_in_row_form_col_major_as_doubles.raw input_dims_as_uint64.raw input_origin_as_doubles.raw input_spacing_as_doubles.raw output_spacing_as_double.raw output_dims_as_uint64.raw output_origin_as_doubles.raw
    // The last two args (output_dims_as_uint64.raw and output_origin_as_doubles.raw) are outputs, the rest are inputs

	// Process the arguments
	if (argc < 8)
	{
		cout << "ERROR: Need seven arguments" << endl;
		return -1 ;
	}
    // Input file names
	string T_in_row_form_file_name(argv[1]) ;
    string input_dims_file_name(argv[2]) ;
    string input_origin_file_name(argv[3]) ;
    string input_spacing_file_name(argv[4]) ;
    string output_spacing_file_name(argv[5]) ;
    // Output file names
    string output_dims_file_name(argv[6]) ;
    string output_origin_file_name(argv[7]) ;

    // Read stuff in
    unique_ptr<double> T_in_row_form(read_col_major_matrix((double*)(0), 4, 4, T_in_row_form_file_name)) ;
    unique_ptr<int64_t> input_dims(read_col_major_matrix((int64_t*)(0), 1, 3, input_dims_file_name)) ;
    unique_ptr<double> input_origin(read_col_major_matrix((double*)(0), 1, 3, input_origin_file_name)) ;
    unique_ptr<double> input_spacing(read_col_major_matrix((double*)(0), 1, 3, input_spacing_file_name)) ;
    unique_ptr<double> output_spacing(read_col_major_matrix((double*)(0), 1, 3, output_spacing_file_name)) ;

    // Allocate outputs
    unique_ptr<int64_t> output_dims(new int64_t[3]) ;
    unique_ptr<double> output_origin(new double[3]) ;

    // Do the transforming
    transform_lattice_3d(
        output_dims.get(), output_origin.get(),
        T_in_row_form.get(),
        input_dims.get(), input_origin.get(), input_spacing.get(),
        output_spacing.get() ) ;

    // Write the outputs to files
    write_col_major_matrix(output_dims_file_name, output_dims, 1, 3) ;
    write_col_major_matrix(output_origin_file_name, output_origin, 1, 3) ;

	// Normal return
    cout << "run_transform_lattice done." << endl;
    return 0;
}
