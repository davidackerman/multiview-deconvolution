#ifndef __IMWARP_FLEXIBLE_H__
#define __IMWARP_FLEXIBLE_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stdbool.h>

#define AFFINE_3D_MATRIX_SIZE 16

int get_number_of_cores() ;

bool is_affine_3d_matrix(const float A[AFFINE_3D_MATRIX_SIZE]);

void print_affine_3d_matrix(const float A[AFFINE_3D_MATRIX_SIZE]);

void imwarp_flexible(const float* input_stack, int64_t input_dims[3], float input_origin[3], float input_spacing[3],
	                 float* output_stack, int64_t output_dims[3], float output_origin[3], float output_spacing[3],
		             float A[AFFINE_3D_MATRIX_SIZE], 
					 bool is_cubic, bool is_background_black) ;

#ifdef __cplusplus
}
#endif

#endif
