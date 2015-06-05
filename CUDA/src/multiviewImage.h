/*
* Copyright (C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  multiviewImage.h
*
*  Created on: June 5th, 2015
*      Author: Fernando Amat
*
* \brief holds a set of pointers to image files
*/

#ifndef __MULTI_VIEW_IMAGE_HEADER_H__
#define __MULTI_VIEW_IMAGE_HEADER_H__

#include <vector>
#include <string>


template<class imgType>
class multiviewImage
{

public:

	multiviewImage();
	multiviewImage(size_t numViews);
	~multiviewImage();

	//I/O functions
	int readImage(const std::string& filename, int pos);//if pos<0 then we add one image to the vector
	static std::string recoverFilenamePatternFromString(const std::string& imgPath, int frame);

	//short IO functions
	size_t getNumberOfViews(){ return imgVec_CPU.size(); };
	void resize(size_t numViews){ imgVec_CPU.resize(numViews, NULL); imgVec_GPU.resize(numViews, NULL); };
	
protected:

private:

	

	std::vector<imgType*> imgVec_CPU;
	std::vector<imgType*> imgVec_GPU;
};

#endif 