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
#include "cuda.h"
#include "book.h"


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
	dimsImgVec.resize(numViews);
};

template<class imgType>
multiviewImage<imgType>::~multiviewImage()
{
	for (size_t ii = 0; ii < imgVec_CPU.size(); ii++)
	{
		if (imgVec_CPU[ii] != NULL)
			delete[]imgVec_CPU[ii];

		if (imgVec_GPU[ii] != NULL)
			HANDLE_ERROR(cudaFree(imgVec_GPU[ii]));
	}
};


//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::deallocateView_CPU(size_t pos)
{
	if (pos < imgVec_CPU.size() && imgVec_CPU[pos] != NULL )
	{
		delete[] imgVec_CPU[pos];
		imgVec_CPU[pos] = NULL;
	}
};

//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::deallocateView_GPU(size_t pos)
{
	if (pos < imgVec_GPU.size() && imgVec_GPU[pos] != NULL)
	{
		HANDLE_ERROR(cudaFree(imgVec_GPU[pos]));
	}
};

//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::allocateView_GPU(size_t pos, size_t numBytes)
{
	if (pos < imgVec_GPU.size())
	{
		if (imgVec_GPU[pos] != NULL)
			deallocateView_GPU(pos);

		HANDLE_ERROR(cudaMalloc((void**)&(imgVec_GPU[pos]), numBytes));
	}
};

//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::allocateView_CPU(size_t pos, size_t numElements)
{
	if (pos < imgVec_CPU.size())
	{
		if (imgVec_CPU[pos] != NULL)
			delete[] imgVec_CPU[pos];

		imgVec_CPU[pos] = new imgType[numElements];
	}
}
//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::setImgDims(size_t pos, const dimsImg &d)
{
	if (pos < dimsImgVec.size())
	{
		dimsImgVec[pos] = d;
	}
}


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
std::int64_t multiviewImage<imgType>::numElements(size_t pos) const
{
	if (pos >= dimsImgVec.size() || dimsImgVec[pos].ndims == 0)
		return 0;

	std::int64_t n = 1;
	for (int ii = 0; ii < dimsImgVec[pos].ndims; ii++)
		n *= dimsImgVec[pos].dims[ii];

	return n;
};
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
		dimsImgVec.push_back(dimsImg());
		pos = imgVec_GPU.size() - 1;
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


	//aupdate dimensions if necessary
	int ndims = 0;
	while (ndims < KLB_DATA_DIMS && imgFull.header.xyzct[ndims] > 1)
	{
		dimsImgVec[pos].dims[ndims] = imgFull.header.xyzct[ndims];
		ndims++;
	}
	dimsImgVec[pos].ndims = ndims;

	return err;
}



//============================================================================
//declare all possible instantitation for the template
template class multiviewImage<uint16_t>;
template class multiviewImage<uint8_t>;
template class multiviewImage<float>;