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

#endif 