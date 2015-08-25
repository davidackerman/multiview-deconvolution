/*
 * weightsBlurryMeasure.h
 *
 *  Created on: Jul 20, 2015
 *      Author: Fernando Amat
 *
 *	\brief: calculate weights for deconvolution using constrat measurement based on shannon entropy of the DCT 8x8 coefficients
 */

#ifndef CUDA_WEIGHTS_BLURRY_MEASURE_H_
#define CUDA_WEIGHTS_BLURRY_MEASURE_H_


#include <stdint.h>

static const float cutoffDCT = 6.0f;
static const float cellDiameterPixels = 15.0f;

//we assume memory for both arrays has been allocated in the GPU
void calculateWeightsDeconvolution(float* weights_CUDA, float* img_CUDA, int64_t *dims, int ndims, float anisotropyZ, bool normalize = true);

#endif /* STANDARDCUDAFUNCTIONS_H_ */
