#ifndef __IMAGE_INTERPOLATION_ML_H__
#define __IMAGE_INTERPOLATION_ML_H__

#include <stdint.h>
#include <stdbool.h>
/* Image and Volume interpolation
 *
 * Function is written by D.Kroon University of Twente (June 2009)
 */

float interpolate_3d_float_gray_ml(double x, double y, double z, int64_t *dims, float *stack, bool is_cubic, bool is_black);

#endif