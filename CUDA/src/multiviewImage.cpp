/*
* Copyright (C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  multiviewImage.cpp
*
*  Created on: June 5th, 2015
*      Author: Fernando Amat
*
* \brief 
*/

#include "multiviewImage.h"
#include "klb_imageIO.h"


using namespace std;


template<class imgType>
multiviewImage<imgType>::multiviewImage()
{
};


template<class imgType>
multiviewImage<imgType>::multiviewImage(size_t numViews)
{
	imgVec_CPU.resize(numViews, NULL);
	imgVec_GPU.resize(numViews, NULL);
};

template<class imgType>
multiviewImage<imgType>::~multiviewImage()
{
	for (size_t ii = 0; ii < imgVec_CPU.size(); ii++)
	{
		if (imgVec_CPU[ii] != NULL)
			delete[]imgVec_CPU[ii];

		if (imgVec_GPU[ii] != NULL)
			delete[]imgVec_GPU[ii];
	}
};

//===========================================================================================

template<class imgType>
string multiviewImage<imgType>::recoverFilenamePatternFromString(const string& imgPath, int frame)
{

	string imgRawPath(imgPath);
	size_t found = imgRawPath.find_first_of("?");
	while (found != string::npos)
	{
		int intPrecision = 0;
		while ((imgRawPath[found] == '?') && found != string::npos)
		{
			intPrecision++;
			found++;
			if (found >= imgRawPath.size())
				break;

		}


		char bufferTM[16];
		switch (intPrecision)
		{
		case 1:
			sprintf(bufferTM, "%.1d", frame);
			break;
		case 2:
			sprintf(bufferTM, "%.2d", frame);
			break;
		case 3:
			sprintf(bufferTM, "%.3d", frame);
			break;
		case 4:
			sprintf(bufferTM, "%.4d", frame);
			break;
		case 5:
			sprintf(bufferTM, "%.5d", frame);
			break;
		case 6:
			sprintf(bufferTM, "%.6d", frame);
			break;
		case 7:
			sprintf(bufferTM, "%.7d", frame);
			break;
		case 8:
			sprintf(bufferTM, "%.28", frame);
			break;
		default:
			cout << "ERROR:recoverFilenamePatternFromString: not prepared for so many ??? in the file pattern" << endl;
		}
		string itoaTM(bufferTM);

		found = imgRawPath.find_first_of("?");
		imgRawPath.replace(found, intPrecision, itoaTM);


		//find next ???
		found = imgRawPath.find_first_of("?");
	}

	return imgRawPath;

}

//===========================================================================================
template<class imgType>
int multiviewImage<imgType>::readImage(const std::string& filename, int pos)
{
	//cout << "=======TODO readImage: we have to read images here!!!====develop a project for image reader wrapper======" << endl;
	
	klb_imageIO imgFull(filename);
	int err = 0;

	err = imgFull.readHeader();
	if (err > 0)
		return err;

	if (sizeof(imgType) != imgFull.header.getBytesPerPixel())
	{
		cout << "ERROR: multiviewImage<imgType>::readImage: class type does not match image type" << endl;
		return 10;
	}

	imgType* imgA = new imgType[imgFull.header.getImageSizePixels()];

	
	err = imgFull.readImageFull((char*)imgA, -1);
	if (err > 0)
		return err;

	if (pos < 0)
	{
		imgVec_CPU.push_back(imgA);
		imgVec_GPU.push_back(NULL);
	}
	else if (pos >= imgVec_CPU.size()){
		cout << "ERROR: multiviewImage<imgType>::readImage: trying to place image in a view that does not exist" << endl;
		delete[] imgA;
		return 12;
	}
	else{
		imgVec_CPU[pos] = imgA;
		imgVec_GPU[pos] = NULL;
	}
	return err;
}



//============================================================================
//declare all possible instantitation for the template
template class multiviewImage<uint16_t>;
template class multiviewImage<uint8_t>;
template class multiviewImage<float>;