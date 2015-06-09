/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  multivieDeconvolution.h
*
*  Created on : June 5th, 2015
* Author : Fernando Amat
*
* \brief main interface to execute multiview deconvolution 
*/

#ifndef __MULTI_VIEW_DECONVOLUTION_IMAGE_HEADER_H__
#define __MULTI_VIEW_DECONVOLUTION_IMAGE_HEADER_H__

#include <string>
#include "multiviewImage.h"


typedef float weightType;
typedef float psfType;
typedef float outputType;

//from cufft.h to avoid including a cuda header here
typedef int cufftHandle;

template<class imgType>
class multiviewDeconvolution
{

public:

	multiviewDeconvolution();	
	~multiviewDeconvolution();


	void setNumberOfViews(int numViews);
	int readImage(const std::string& filename, int pos, const std::string& type);
	//perfoms all the preallocation and precalculation for the deconvolution
	int allocate_workspace();
	
	//different deconvolution methods
	void deconvolution_LR_TV(int numIters, float lambdaTV);//lucy-richardson with totalvariation regularization

protected:	


	//main objects to hold values
	multiviewImage<weightType> weights;
	multiviewImage<psfType> psf;
	multiviewImage<imgType> img;

	multiviewImage<outputType> J;//holds the output, so it is going to have a different number of views

	//holds the fft plans 
	cufftHandle fftPlanFwd, fftPlanInv;

private:
};

#endif 