/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  binning3Darray_mex.cpp
*
*  Created on : August 3rd, 2015
* Author : Fernando Amat
*
* \brief small mex file to bin an image by 2 by averaging neighboring pixels. Using imfilter takes too much memory due to copies
*/

#include "mex.h"
#include <stdint.h>

using namespace std;

template<class T>
void binImage(const T* img, T* imgBin, const int64_t dims[3])
{
	const T neighSize = 8;
	const int64_t strideX = dims[0];
	const int64_t strideXY = dims[0] * dims[1];


	int64_t idxBin = 0;
	int64_t offset;
	for (int64_t zz = 0; zz < dims[2] / 2; zz++)
	{
		for (int64_t yy = 0; yy < dims[1] / 2; yy++)
		{
			offset = 2 * dims[0] * (yy + dims[1] * zz);
			for (int64_t xx = 0; xx < dims[0] / 2; xx++)
			{
				//int64_t offset = 2 * ( xx + dims[0] * (yy + dims[1] * zz) );
				imgBin[idxBin++] = (img[offset] + img[offset + 1] + img[offset + strideX] + img[offset + strideX + 1] + img[offset + strideXY] + img[offset + 1 + strideXY] + img[offset + strideX + strideXY] + img[offset + strideX + 1 + strideXY]) / neighSize;
				offset += 2;
			}
		}
	}
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	
	/* Check for proper number of arguments. */
	if (nrhs != 1) 
	{
		mexErrMsgTxt("Function requires 1 input");
	}

	//get dimensions
	mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
	if (ndims != 3)
	{
		mexErrMsgTxt("Functions only accepts 3D arrays");
	}

	const mwSize *dimsM = mxGetDimensions(prhs[0]);
	int64_t dims[3];
	mwSize dimsBin[3];
	for (int ii = 0; ii < ndims; ii++)
	{
		dims[ii] = dimsM[ii];
		dimsBin[ii] = (dims[ii] / 2);

	}

	if (nlhs > 1)
	{
		mexErrMsgTxt("Only one output");
	}
	else if (nlhs == 1)
	{
		/* Create output array */
		//setting data type
		switch (mxGetClassID(prhs[0]))
		{
		case mxUINT8_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxUINT8_CLASS, mxREAL);			
			binImage((uint8_t*)(mxGetData(prhs[0])), (uint8_t*)(mxGetData(plhs[0])), dims);
			break;

		case mxUINT16_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxUINT16_CLASS, mxREAL);
			binImage((uint16_t*)(mxGetData(prhs[0])), (uint16_t*)(mxGetData(plhs[0])), dims);
			break;

		case mxUINT32_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxUINT32_CLASS, mxREAL);
			binImage((uint32_t*)(mxGetData(prhs[0])), (uint32_t*)(mxGetData(plhs[0])), dims);
			break;

		case mxUINT64_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxUINT64_CLASS, mxREAL);
			binImage((uint64_t*)(mxGetData(prhs[0])), (uint64_t*)(mxGetData(plhs[0])), dims);
			break;

		case mxINT8_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxINT8_CLASS, mxREAL);
			binImage((int8_t*)(mxGetData(prhs[0])), (int8_t*)(mxGetData(plhs[0])), dims);
			break;

		case mxINT16_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxINT16_CLASS, mxREAL);
			binImage((int16_t*)(mxGetData(prhs[0])), (int16_t*)(mxGetData(plhs[0])), dims);
			break;

		case mxINT32_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxINT32_CLASS, mxREAL);
			binImage((int32_t*)(mxGetData(prhs[0])), (int32_t*)(mxGetData(plhs[0])), dims);
			break;

		case mxINT64_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxINT64_CLASS, mxREAL);
			binImage((int64_t*)(mxGetData(prhs[0])), (int64_t*)(mxGetData(plhs[0])), dims);
			break;

		case mxSINGLE_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxSINGLE_CLASS, mxREAL);
			binImage((float*)(mxGetData(prhs[0])), (float*)(mxGetData(plhs[0])), dims);
			break;

		case mxDOUBLE_CLASS:
			plhs[0] = mxCreateNumericArray(3, dimsBin, mxDOUBLE_CLASS, mxREAL);
			binImage((double*)(mxGetData(prhs[0])), (double*)(mxGetData(plhs[0])), dims);
			break;

		default:
			mexErrMsgTxt("Data type not supported");
		}
				
	}
}
