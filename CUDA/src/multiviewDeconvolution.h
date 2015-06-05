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
* \brief main interface to execute multiview deconvolution (it has ome abstract methods)
*/

#ifndef __MULTI_VIEW_DECONVOLUTION_IMAGE_HEADER_H__
#define __MULTI_VIEW_DECONVOLUTION_IMAGE_HEADER_H__

#include "multiviewImage.h"


typedef float weightType;
typedef float psfType;
typedef float outputType;

template<class imgType>
class multiviewDeconvolution
{

public:

	multiviewDeconvolution();	
	~multiviewDeconvolution();
	
protected:

	//implements main deconvolution ste using variables
	virtual void deconvolution(int numIters);


	//main objects to hold values
	multiviewImage<weightType> weights;
	multiviewImage<psfType> psf;
	multiviewImage<imgType> img;

	outputType* J;


private:
};

#endif 