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

// What is described by "[a00,a10, a20, a30,a01,a11,a21,a31,...]" is *column*-major order, which is what Matlab uses.
// Having the translation in the last row is indeed what Matlab's imwarp() + affine3d() use.  
// In particular, output_image = imwarp(input_image, affine3d(A)) uses the transform output_image_coord = input_image_coord * A, with output_coord 
// and input_coord being row vectors of the form [x y z 1].  -- ALT, 2016-11-21

//auxiliary functions for affine matrices
void affine_3d_transpose(const float  A[AFFINE_3D_MATRIX_SIZE], float Atr[AFFINE_3D_MATRIX_SIZE]);
void affine_3d_compose(const float  A[AFFINE_3D_MATRIX_SIZE], const float B[AFFINE_3D_MATRIX_SIZE], float C[AFFINE_3D_MATRIX_SIZE]);  // C = A * B
  // This yields the equivalent transform to transforming by A, then transforming by B, 
  // if A and B are both the row-form representations of the respective transforms.
  // i.e. the form where output_coord = input_coord * A, with with output_coord 
  // and input_coord being row vectors of the form [x y z 1].  -- ALT, 2016-11-21

void affine_3d_inverse(const float  A[AFFINE_3D_MATRIX_SIZE], float Ainv[AFFINE_3D_MATRIX_SIZE]);
bool affine_3d_isAffine(const float A[AFFINE_3D_MATRIX_SIZE]);

void affine3d_printMatrix(const float A[AFFINE_3D_MATRIX_SIZE]);


#ifdef __cplusplus
}
#endif

#endif