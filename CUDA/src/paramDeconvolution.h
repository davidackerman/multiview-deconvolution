/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  paramDeconvolution.h
*
*  Created on : June 11th, 2015
* Author : Fernando Amat
*
* \brief class storing all possible parameters for different deconvolution algorithms 
*/

#ifndef __PARAM_DECONVOLUTION_HEADER_H__
#define __PARAM_DECONVOLUTION_HEADER_H__

#include <string>

typedef float imgTypeDeconvolution;//type of input image (I could use a template later if needed)

struct paramDeconvolution
{
	std::string filepath;
	std::string filePatternPSF;
	std::string filePatternWeights;
	std::string filePatternImg;


	int numIters;
	int Nviews;
	
    
    imgTypeDeconvolution imgBackground;	
	float lambdaTV;	

};

#endif 