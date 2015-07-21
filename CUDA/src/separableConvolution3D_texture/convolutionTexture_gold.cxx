/*
* Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation and 
* any modifications thereto.  Any use, reproduction, disclosure, or distribution 
* of this software and related documentation without an express license 
* agreement from NVIDIA Corporation is strictly prohibited.
* 
*/



#include "convolutionTexture_common.h"



////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowsCPU(
                                   float *h_Dst,
                                   float *h_Src,
                                   float *h_Kernel,
                                   int imageW,
                                   int imageH,
                                   int imageD,
                                   int kernelR
                                   )
{
    for(int z = 0; z < imageD; z++)
    {
        for(int y = 0; y < imageH; y++)
        {
            for(int x = 0; x < imageW; x++)
            {
                float sum = 0;
                for(int k = -kernelR; k <= kernelR; k++)
                {
                    int d = x + k;
                    if(d < 0) d = 0;
                    if(d >= imageW) d = imageW - 1;
                    sum += h_Src[(imageW * imageH * z) + (y * imageW) + d] * h_Kernel[kernelR - k];
                }
                h_Dst[(imageW * imageH * z) + (y * imageW) + x] = sum;
            }
        }
    }
}




////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionColumnsCPU(
                                      float *h_Dst,
                                      float *h_Src,
                                      float *h_Kernel,
                                      int imageW,
                                      int imageH,
                                      int imageD,
                                      int kernelR
                                      )
{
    for(int z = 0; z < imageD; z++)
    {
        for(int x = 0; x < imageW; x++)
        { 
            for(int y = 0; y < imageH; y++)
            {
                float sum = 0;
                for(int k = -kernelR; k <= kernelR; k++)
                {
                    int d = y + k;
                    if(d < 0) d = 0;
                    if(d >= imageH) d = imageH - 1;
                    sum += h_Src[(imageW * imageH * z) + (d * imageW) + x] * h_Kernel[kernelR - k];
                }
                h_Dst[(imageW * imageH * z) + (y * imageW) + x] = sum;
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Reference depth convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionDepthCPU(
                                    float *h_Dst,
                                    float *h_Src,
                                    float *h_Kernel,
                                    int imageW,
                                    int imageH,
                                    int imageD,
                                    int kernelR
                                    )
{
    for(int y = 0; y < imageH; y++)
    {
        for(int x = 0; x < imageW; x++)
        {

            for(int z = 0; z < imageD; z++)
            {
                float sum = 0;
                for(int k = -kernelR; k <= kernelR; k++)
                {
                    int d = z + k;
                    if(d < 0) d = 0;
                    if(d >= imageD) d = imageD - 1;
                    sum += h_Src[(imageW * imageH * d) + (y * imageW) + x] * h_Kernel[kernelR - k];
                }
                h_Dst[(imageW * imageH * z) + (y * imageW) + x] = sum;
            }
        }
    }
}
