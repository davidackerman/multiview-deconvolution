#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "image_interpolation_ml.h"
#include "multiple_os_thread.h"

#define AFFINE_3D_MATRIX_SIZE 16

static __inline int64_t mindex3(int64_t i_x, int64_t i_y, int64_t i_z, int64_t n_x, int64_t n_y) { return i_z*n_x*n_y + i_x*n_y + i_y ; }  ;

bool is_affine_3d_matrix(const float A[AFFINE_3D_MATRIX_SIZE])
{
	// A is assumed to be stored col-major, and to be a "row form" affine transform matrix, s.t. y' = x'*A .
	// Check the last column. It should be [0, 0, 0, 1]'
	return (A[12] == 0.0f && A[13] == 0.0f && A[14] == 0.0f && A[15] == 1.0f) ;
}

void print_affine_3d_matrix(const float A[AFFINE_3D_MATRIX_SIZE])
{
	// A is assumed to be stored col-major, and to be a "row form" affine transform matrix, s.t. y' = x'*A .
	int ii;
	for (ii = 0; ii < 4; ii++)
		printf("%.6f\t%.6f\t%.6f\t%.6f\n", A[ii + 4 * 0], A[ii + 4 * 1], A[ii + 4 * 2], A[ii + 4 * 3]);
}

// Macro to get the element of A at row i, col j, assuming A is stored col-major
#define _m(i,j) (A[i + 4 * j])
// Macro to get the element of Ainv at row i, col j, assuming Ainv is stored col-major
#define _minv(i,j) (Ainv[i + 4 * j])
void affine_3d_inverse(const float  A[AFFINE_3D_MATRIX_SIZE], float Ainv[AFFINE_3D_MATRIX_SIZE])
{
	// A is assumed to be stored col-major, and to be a "row form" affine transform matrix, s.t. y' = x'*A .
	if (!is_affine_3d_matrix(A))
	{
		printf("ERROR: affine_3d_inverse: trying to calculate inverse of a non-affine matrix\n");
		print_affine_3d_matrix(A);
		exit(2);
	}

	//A = [ [R , zeros(3,1)]; [Tx Ty Tz 1] ]
	//Ainv = [ [Rinv , zeros(3,1)]; [-[Tx Ty Tz]*Rinv 1] ]




	// computes the inverse of a matrix m
	double det = _m(0, 0) * (_m(1, 1) * _m(2, 2) - _m(2, 1) * _m(1, 2)) -
		_m(0, 1) * (_m(1, 0) * _m(2, 2) - _m(1, 2) * _m(2, 0)) +
		_m(0, 2) * (_m(1, 0) * _m(2, 1) - _m(1, 1) * _m(2, 0));

	if (fabs(det) < 1e-8)
	{
		printf("ERROR: affine_3d_inverse: matrix is close to singular\n");
		print_affine_3d_matrix(A);
		exit(2);
	}

	double invdet = 1 / det;

	_minv(0, 0) = (float) ((_m(1, 1) * _m(2, 2) - _m(2, 1) * _m(1, 2)) * invdet) ;
    _minv(0, 1) = (float) ((_m(0, 2) * _m(2, 1) - _m(0, 1) * _m(2, 2)) * invdet) ;
    _minv(0, 2) = (float) ((_m(0, 1) * _m(1, 2) - _m(0, 2) * _m(1, 1)) * invdet) ;
    _minv(1, 0) = (float) ((_m(1, 2) * _m(2, 0) - _m(1, 0) * _m(2, 2)) * invdet) ;
    _minv(1, 1) = (float) ((_m(0, 0) * _m(2, 2) - _m(0, 2) * _m(2, 0)) * invdet) ;
    _minv(1, 2) = (float) ((_m(1, 0) * _m(0, 2) - _m(0, 0) * _m(1, 2)) * invdet) ;
    _minv(2, 0) = (float) ((_m(1, 0) * _m(2, 1) - _m(2, 0) * _m(1, 1)) * invdet) ;
    _minv(2, 1) = (float) ((_m(2, 0) * _m(0, 1) - _m(0, 0) * _m(2, 1)) * invdet) ;
    _minv(2, 2) = (float) ((_m(0, 0) * _m(1, 1) - _m(1, 0) * _m(0, 1)) * invdet) ;


	//translation for bottom row
	Ainv[3] = -(A[3] * _minv(0, 0) + A[7] * _minv(1, 0) + A[11] * _minv(2, 0));
	Ainv[7] = -(A[3] * _minv(0, 1) + A[7] * _minv(1, 1) + A[11] * _minv(2, 1));
	Ainv[11] = -(A[3] * _minv(0, 2) + A[7] * _minv(1, 2) + A[11] * _minv(2, 2));

	//translation for last column
	Ainv[12] = -(A[12] * _minv(0, 0) + A[13] * _minv(0, 1) + A[14] * _minv(0, 2));
	Ainv[13] = -(A[12] * _minv(1, 0) + A[13] * _minv(1, 1) + A[14] * _minv(1, 2));
	Ainv[14] = -(A[12] * _minv(2, 0) + A[13] * _minv(2, 1) + A[14] * _minv(2, 2));

	//set last element	
	Ainv[15] = 1.0f;
}

//================================================================================================================
//from http://www.cprogramming.com/snippets/source-code/find-the-number-of-cpu-cores-for-windows-mac-or-linux
int getNumberOfCores()
{
#ifdef WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	return sysinfo.dwNumberOfProcessors;
#elif MACOS
	int nm[2];
	size_t len = 4;
	uint32_t count;

	nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
	sysctl(nm, 2, &count, &len, NULL, 0);

	if (count < 1) {
		nm[1] = HW_NCPU;
		sysctl(nm, 2, &count, &len, NULL, 0);
		if (count < 1) { count = 1; }
	}
	return count;
#else
	return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

//================================================================================================================
voidthread transform_subvolume(void **args) {
	// Sort out the arguments
	int64_t* output_dims = (int64_t *)(args[0]);
	float* output_origin = (float *)(args[1]);
	float* output_spacing = (float *)(args[2]);
	float* Ainv = (float*)(args[3]);
	float* input_stack = args[4];
	float* output_stack = args[5];
	int thread_id = *((int *)(args[6])) ;
	bool is_cubic = *((bool *)(args[7]));
	bool is_background_black = *((bool *)(args[8]));
	int n_threads = *((int *)(args[9]));
	int64_t* input_dims = args[10] ;
	float* input_origin = args[11] ;
	float* input_spacing = (float *)(args[12]);

	// Decode the interpolation_mode
	//int is_black = (interpolation_mode == 1 || interpolation_mode == 3) ;
	//int is_cubic = (interpolation_mode == 2 || interpolation_mode == 3) ;

	float acomp0 = Ainv[3];  // the x translation
	float acomp1 = Ainv[7];  // the y translation 
	float acomp2 = Ainv[11];  // the z translation
	//  Loop through all output image pixel coordinates */
	for (int64_t i_z_out = thread_id; i_z_out<output_dims[2]; i_z_out = i_z_out + n_threads) {
		float z_out = output_spacing[2] * (i_z_out + 0.5f) + output_origin[2] ;
		float bcomp0 = Ainv[2] * z_out + acomp0;
		float bcomp1 = Ainv[6] * z_out + acomp1;
		float bcomp2 = Ainv[10] * z_out + acomp2;
		for (int64_t i_y_out = 0; i_y_out<output_dims[0]; i_y_out++) {
			float y_out = output_spacing[1] * (i_y_out + 0.5f) + output_origin[1] ;
			float ccomp0 = Ainv[1] * y_out + bcomp0;
			float ccomp1 = Ainv[5] * y_out + bcomp1;
			float ccomp2 = Ainv[9] * y_out + bcomp2;
			for (int64_t i_x_out = 0; i_x_out<output_dims[1]; i_x_out++) {
				float x_out = output_spacing[0] * (i_x_out + 0.5f) + output_origin[0] ;
				float x_in = Ainv[0] * x_out + ccomp0;
				float y_in = Ainv[4] * x_out + ccomp1;
				float z_in = Ainv[8] * x_out + ccomp2;

				float i_x_in = (x_in - input_origin[0])/input_spacing[0] - 0.5f ;  // NB: Not necessarily integral.
				float i_y_in = (y_in - input_origin[1])/input_spacing[1] - 0.5f ;  // NB: Not necessarily integral.
				float i_z_in = (z_in - input_origin[2])/input_spacing[2] - 0.5f ;  // NB: Not necessarily integral.

				int64_t i_out = mindex3(i_x_out, i_y_out, i_z_out, output_dims[1], output_dims[0]);

				output_stack[i_out] = interpolate_3d_float_gray(i_x_in, i_y_in, i_z_in, input_dims, input_stack, is_cubic, is_background_black);
			}
		}
	}

	//  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
	EndThread;
}

//================================================================================================================
void imwarp_flexible(const float* input_stack, int64_t input_dims[3], float input_origin[3], float input_spacing[3],
	                 float* output_stack, int64_t output_dims[3], float output_origin[3], float output_spacing[3], 
					 float A[AFFINE_3D_MATRIX_SIZE], 
					 bool is_cubic, bool is_background_black)
{
	// This assumes imIn is stored Matlab-style in memory: It's a 3D array with dims [n_y n_x n_z], stored col-major.
	// On exit, imOut will be in the same form.
	// A is assumed to represent a 4x4 affine transform matrix, stored col-major.  
	// A should be the "row form" representation of the 3D affine transform, i.e the last column should be [0 0 0 1]', and the last
	// row should represent the translation.  In this form, output_coord = input_coord * A, with output_coord and input_coord being 
	// row vectors of the form [x y z 1].
	// dimsIn gives the dimensions of the input stack, in the order n_y, n_x, n_z.  (sic)
	// dimsOut gives the desired dimensions of the output stack, in the order n_y, n_x, n_z.
	// Each element of dimsOut must be greater than or equal to the corresponding element of dimsIn.
	// The input stack is padded with zeros on the "high" end of each dimension out to size dimsOut before the transforming is done.
	// Once the input stack is padded out to the same dimensions as the output stack, the affine transform is done.
	// It is done s.t. it is equal to the Matlab commands:
	//
	// default_frame = ...
	//     imref3d(size(input_stack), ...
	//	           0.5 + [0 size(input_stack, 2)], ...
	//	           0.5 + [0 size(input_stack, 1)], ...
	//	           0.5 + [0 size(input_stack, 3)]);
	//
	// output_stack = imwarp(input_stack, default_frame, ...
	//	                     affine3d(A), ...
	//	                     'OutputView', default_frame);
	//
	// That is, the element at row i, col j, page k of the input stack (using zero-based indexing) is assumed to be located at 
	// <x,y,z> coordinate <j+1, i+1, k+1> (note that cols map to x, rows to y), and the same for the output stack.  The equation
	// output_coord = input_coord * A then relates the output <x,y,z> coordinates to the input <x,y,z> coordinates, with 
	// interpolation used when a given output point does not correspond exactly to an on-grid input point.

	// Take the inverse of A once, since we need it at each voxel
	float Ainv[AFFINE_3D_MATRIX_SIZE];
	affine_3d_inverse(A, Ainv);

	// Determine the number of threads to use
	int n_threads = getNumberOfCores();

	// Reserve room for handles of threads in ThreadList  
	ThreadHANDLE* thread_list = (ThreadHANDLE*)malloc(n_threads*sizeof(ThreadHANDLE));

	// Define an array of thread IDs
	int* thread_ids = (int *)malloc(n_threads*sizeof(int));  // An array of thread IDs
	for (int i = 0; i < n_threads; i++)
		thread_ids[i] = i;

	// Define an array of pointers to thread argument arrays
	const int n_thread_args = 13 ;
	void*** thread_args = (void ***)malloc(n_threads*sizeof(void **));
	for (int i = 0; i < n_threads; i++)
		thread_args[i] = (void **)malloc(n_thread_args * sizeof(void *));

	// Package thread arguments
	for (int i = 0; i < n_threads; i++)  {
		thread_args[i][ 0] = (void *)output_dims;
		thread_args[i][ 1] = (void *)output_origin;
		thread_args[i][ 2] = (void *)output_spacing;
		thread_args[i][ 3] = (void *)Ainv;
		thread_args[i][ 4] = (void *)input_stack;
		thread_args[i][ 5] = (void *)output_stack;
		thread_args[i][ 6] = (void *)(&(thread_ids[i]));
		thread_args[i][ 7] = (void *)(&is_cubic);
		thread_args[i][ 8] = (void *)(&is_background_black);
		thread_args[i][ 9] = (void *)(&n_threads);
		thread_args[i][10] = (void *)input_dims;
		thread_args[i][11] = (void *)input_origin;
		thread_args[i][12] = (void *)input_spacing;
	}

	// Start the threads
	for (int i = 0; i < n_threads; i++)
		StartThread(thread_list[i], &transform_subvolume, thread_args[i])

	// Wait for all the threads to finish
	for (int i = 0; i < n_threads; i++)  {
		WaitForThreadFinish(thread_list[i]);
	}

	// Free allocated memory
	for (int i = 0; i<n_threads; i++)
		free(thread_args[i]);
	free(thread_args);
	free(thread_ids);
	free(thread_list);
	//free(Nthreadsf);
	//free(moded);
}

