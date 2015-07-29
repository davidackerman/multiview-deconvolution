#ifndef __AFFINE_TRANSFORM_3D_SINGLE_H__
#define __AFFINE_TRANSFORM_3D_SINGLE_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stdbool.h>

#define AFFINE_3D_MATRIX_SIZE 16

/* This function transforms a volume with a 4x4 transformation matrix
 *
 * affine transform Iout=affine_transform_3d_double(Iin,Minv,mode)
 *
 * inputs,
 *  Iin: The greyscale 3D input image
 *  Minv: The (inverse) 4x4 transformation matrix
 *  mode: If 0: linear interpolation and outside pixels set to nearest pixel
 *           1: linear interpolation and outside pixels set to zero
 *           2: cubic interpolation and outsite pixels set to nearest pixel
 *           3: cubic interpolation and outside pixels set to zero
 * output,
 *  Iout: The transformed 3D image
 *
 *
 * 
 * Function is written by D.Kroon University of Twente (June 2009)
 */


//this function allows to call it using the same exact elements as the equivalent Matlab call
void imwarpFast_MatlabEquivalent(const float* imIn, float* imOut, int64_t dimsIn[3], int64_t dimsOut[3], float A[AFFINE_3D_MATRIX_SIZE], int interpMode);
void affineTransform_3d_float(const float* imIn, float* imOut, int64_t dims[3], float A[AFFINE_3D_MATRIX_SIZE], int interpMode);

//auxiliary functions
int getNumberOfCores();
float* fa_padArrayWithZeros(const float* im, const int64_t *dimsNow, const int64_t *dimsAfterPad, int ndims);




//the same as Matlab: row major order A[AFFINE_3D_MATRIX_SIZE] = [a00,a10, a20, a30,a01,a11,a21,a31,...] with a30 = tx; 
//auxiliary functions for affine matrices
void affine_3d_transpose(const float  A[AFFINE_3D_MATRIX_SIZE], float Atr[AFFINE_3D_MATRIX_SIZE]);
void affine_3d_compose(const float  A[AFFINE_3D_MATRIX_SIZE], const float B[AFFINE_3D_MATRIX_SIZE], float C[AFFINE_3D_MATRIX_SIZE]);//C = A * B
void affine_3d_inverse(const float  A[AFFINE_3D_MATRIX_SIZE], float Ainv[AFFINE_3D_MATRIX_SIZE]);
bool affine_3d_isAffine(const float A[AFFINE_3D_MATRIX_SIZE]);

void affine3d_printMatrix(const float A[AFFINE_3D_MATRIX_SIZE]);


#ifdef __cplusplus
}
#endif

#endif