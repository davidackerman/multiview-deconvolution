/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  imgUtils.h
*
*  Created on : June 11th, 2015
* Author : Fernando Amat
*
* \brief random functions that are generically useful 
*/

#ifndef __FA_IMG_UTILS_HEADER_H__
#define __FA_IMG_UTILS_HEADER_H__

#include <cstdint>
#include <string>

template<class imgType>
imgType* fa_padArrayWithZeros(const imgType* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);

std::string generateTempFilename(const char* prefix);

int copySlice(float* target, int64_t* targetSize,
              int64_t* nElementsToTrim,
              int64_t arity, int64_t sliceArity,
              float* source, int64_t* sourceSize) ;
int64_t indexOfFirstSuperthresholdSlice(float *x, int64_t arity, int64_t* xDims, int64_t iDim, float threshold) ;
int64_t indexOfLastSuperthresholdSlice(float *x, int64_t arity, int64_t* xDims, int64_t iDim, float threshold) ;
bool isSliceSuperthreshold(float *x, int64_t arity, int64_t* xDims, int64_t iDim, int64_t iElement, float threshold) ;
int64_t chunkSize(int64_t* xDims, int64_t iDim) ;
int64_t chunkCount(int64_t arity, int64_t* xDims, int64_t iDim) ;
int64_t elementsPerSlice(int64_t arity, int64_t* size, int64_t sliceArity) ;

void transform_lattice_3d(int64_t* targetCount, double* targetOrigin,
                          float* A,
                          int64_t* sourceCount, double* sourceOrigin, double* sourceSpacing,
                          double * targetSpacing) ;
void transform_cuboid_3d(double* targetOrigin, double* targetExtent,
                         float* A,
                         double* sourceOrigin, double* sourceExtent) ;
void lattice_from_cuboid_3d(int64_t* count, double* origin,
                            double* ideal_origin, double* ideal_extent, double *spacing) ;
void affine_transform_3d(double* y, float* T, double* x) ;
void extent_from_dims_and_spacing_3d(double* extent, int64_t* dims, double* spacing) ;
void elementwise_product_3d(double* z, double* x, double* y) ;
void scalar_product_3d(double* y, double a, double* x) ;
void elementwise_quotient_3d(double* z, double* x, double* y) ;
void dims_from_extent_with_unit_spacing_3d(int64_t* y, double* x) ;
void difference_3d(double* z, double* x, double* y) ;
void sum_3d(double* z, double* x, double* y) ;
void sum_in_place_3d(double* y, double* x) ;
int64_t element_count_from_dims_3d(int64_t* dims) ;
float normalize_in_place_3d(float* x, int64_t* dims) ;
void pos_in_place_3d(float *x, int64_t dims[3]) ;
int64_t find_first_nonzero_element_3d(float *x, int64_t dims[3]) ;

#endif 
