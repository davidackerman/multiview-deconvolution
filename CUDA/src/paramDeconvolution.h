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
#include <vector>

typedef float imgTypeDeconvolution;//type of input image (I could use a template later if needed)

struct paramDeconvolution
{
	std::string filepath;
	std::string filePatternPSF;
	std::string filePatternWeights;
	std::string filePatternImg;

	//in case each view has individual information (reading from XML)
	std::vector<std::string> fileImg;
	std::vector<std::string> filePSF;
	std::vector< std::vector<float> > Acell;//to store affine transformations for each view
    std::vector< std::vector<double> > AcellAsDouble;//to store affine transformations for each view, as doubles

	int verbose;

	int numIters;
	int Nviews;
	float anisotropyZ;
	
	int blockZsize;

	bool saveAsUINT16;
	struct {			       // when saveAsUINT16=true, and min/max saturation values are set
		float min, max;        // the min will be mapped to 0 and max will be mapped to 2^16-1.
		bool use_min, use_max; // Otherwise, the min and max values used will be the min and max
	} saturation;              // values found in the output volume.
	float minSaturation; 
	float maxSaturation;
	float weightThr;

    imgTypeDeconvolution imgBackground;	
	float lambdaTV;

	std::string outputFilePrefix;

    bool isPSFAlreadyTransformed ;

	//default parameters
	void setDefaultParam()
	{
		numIters = 40;
		lambdaTV = 0.0001f;
		imgBackground = 100.0f;
		verbose = 0;
		blockZsize = -1;//compute deconvolution all at once
		outputFilePrefix = std::string("");
		saveAsUINT16 = true;
		weightThr = 0.0f;

		saturation.use_max = false;
		saturation.use_min = false;

        isPSFAlreadyTransformed = true ;
	}

	float getAnisotropyZfromAffine();

};

#endif 