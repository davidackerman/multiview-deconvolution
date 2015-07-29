/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  paramDeconvolution.cpp
*
*  Created on : June 11th, 2015
* Author : Fernando Amat
*
* \brief class storing all possible parameters for different deconvolution algorithms
*/

#include "paramDeconvolution.h"


float paramDeconvolution::getAnisotropyZfromAffine()
{
	float Id[16] = { 1.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 1.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, -1.0f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 1.000000000000f };//identity matrix except in Z location

	int jj;
	for (size_t ii = 0; ii < Acell.size(); ii++)
	{	
		for (jj = 0; jj < 16; jj++)
		{
			if (Id[jj] > -0.5f && fabs(Id[jj] - Acell[ii][jj]) > 1e-3)
				break;//this is not a match
		}

		if (jj == 16)//it pass the test
			return Acell[ii][10];
	}

	return 1.0f;
}