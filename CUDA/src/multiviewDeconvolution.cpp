/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  multivieDeconvolution.cpp
*
*  Created on : June 5th, 2015
* Author : Fernando Amat
*
* \brief main interface to execute multiview deconvolution (it has ome abstract methods)
*/

#include <cstdint>
#include <iostream>
#include "multiviewDeconvolution.h"

using namespace std;


template<class imgType>
multiviewDeconvolution<imgType>::multiviewDeconvolution()
{
	J.resize(1);//allocate for the output
}

//=======================================================
template<class imgType>
multiviewDeconvolution<imgType>::~multiviewDeconvolution()
{
}


//=======================================================
template<class imgType>
void multiviewDeconvolution<imgType>::setNumberOfViews(int numViews)
{
	weights.resize(numViews);
	psf.resize(numViews);
	img.resize(numViews);
}

//=======================================================
template<class imgType>
int multiviewDeconvolution<imgType>::readImage(const std::string& filename, int pos, const std::string& type)
{
	if (type.compare("weight") == 0)
		return weights.readImage(filename, pos);
	else if (type.compare("psf") == 0)
		return psf.readImage(filename, pos);
	else if (type.compare("img") == 0)
		return img.readImage(filename, pos);
	
	cout << "ERROR: multiviewDeconvolution<imgType>::readImage :option " << type << " not recognized" << endl;
	return 3;
}

//=======================================================
template<class imgType>
int multiviewDeconvolution<imgType>::allocate_workspace()
{
	cout << "===================TODO=================" << endl;
	return 0;
}


//=======================================================
template<class imgType>
void multiviewDeconvolution<imgType>::deconvolution_LR_TV(int numIters, float lambdaTV)
{
	cout << "===================TODO=================" << endl;
}


//=================================================================
//declare all possible instantitation for the template
template class multiviewDeconvolution<uint16_t>;
template class multiviewDeconvolution<uint8_t>;
template class multiviewDeconvolution<float>;