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

#endif 
