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
#include "klb_Cwrapper.h"
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
void multiviewImage<imgType>::copyView_GPU_to_CPU(size_t pos)
{
	if (pos < imgVec_CPU.size() && imgVec_GPU[pos] != NULL)
	{
		HANDLE_ERROR( cudaMemcpy(imgVec_CPU[pos], imgVec_GPU[pos], numElements(pos) * sizeof(imgType), cudaMemcpyDeviceToHost) );
	}
}

//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::copyView_CPU_to_GPU(size_t pos)
{
	if (pos < imgVec_CPU.size() && imgVec_GPU[pos] != NULL)
	{
		HANDLE_ERROR( cudaMemcpy(imgVec_GPU[pos], imgVec_CPU[pos], numElements(pos) * sizeof(imgType), cudaMemcpyHostToDevice) );
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
int multiviewImage<imgType>::getImageDimensionsFromHeader(const std::string& filename, std::uint32_t xyzct[MAX_DATA_DIMS])
{
    //read klb header
	uint32_t	xyzctR[KLB_DATA_DIMS];
	uint32_t	blockSizeR[KLB_DATA_DIMS];
	char metadataR[KLB_METADATA_SIZE];
	enum KLB_DATA_TYPE dataTypeR;
	enum KLB_COMPRESSION_TYPE compressionTypeR;
	float32_t pixelSizeR[KLB_METADATA_SIZE];


	int err = readKLBheader(filename.c_str(), xyzctR, &dataTypeR, pixelSizeR, blockSizeR, &compressionTypeR, metadataR);

	if (err > 0)
		return err;

	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		xyzct[ii] = xyzctR[ii];
	return 0;
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
int multiviewImage<imgType>::readROI(const std::string& filename, int pos, const klb_ROI& ROI)
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
	
	imgType* imgA = new imgType[ROI.getSizePixels()];
	
	err = imgFull.readImage((char*)imgA, &ROI, -1);
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
		if (imgVec_CPU[pos] != NULL)
			delete[] (imgVec_CPU[pos]);//TODO: if it was the same size, we do not need to reallocate

		imgVec_CPU[pos] = imgA;		
	}


	//aupdate dimensions if necessary
	int ndims = 0;
	while (ndims < KLB_DATA_DIMS && ROI.getSizePixels(ndims) > 1)
	{
		dimsImgVec[pos].dims[ndims] = ROI.getSizePixels(ndims);
		ndims++;
	}
	dimsImgVec[pos].ndims = ndims;

	return err;
}
//===========================================================================================
template<class imgType>
int multiviewImage<imgType>::readImage(const std::string& filename, int pos)
{

	klb_imageIO imgFull(filename);
	int err = 0;

	err = imgFull.readHeader();
	if (err > 0)
		return err;

	klb_ROI ROI;
	ROI.defineFullImage(imgFull.header.xyzct);

	return readROI(filename, pos, ROI);
}
//===========================================================================================
template<class imgType>
int multiviewImage<imgType>::writeImage(const std::string& filename, int pos)
{
	if ( pos >= imgVec_CPU.size() || imgVec_CPU[pos] == NULL)
		return 0;

	//initialize I/O object	
	klb_imageIO imgIO(filename);

	uint32_t xyzct[KLB_DATA_DIMS];
	for (int ii = 0; ii < dimsImgVec[pos].ndims; ii++)
	{
		xyzct[ii] = dimsImgVec[pos].dims[ii];
	}
	for (int ii = dimsImgVec[pos].ndims; ii <KLB_DATA_DIMS; ii++)
	{
		xyzct[ii] = 1;
	}

	//set header
	switch (sizeof(imgType))//TODO: this is not accurate since int8 will be written as uint8
	{
	case 1:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT8_TYPE);
		break;
	case 2:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT16_TYPE);
		break;
	case 4:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::FLOAT32_TYPE);
		break;
	default:
		cout << "ERROR: format not supported yet" << endl;
		return 10;
	}	

	//write image
	int error = imgIO.writeImage((char*)(imgVec_CPU[pos]), -1);//all the threads available

	if (error > 0)
	{
		switch (error)
		{
		case 2:
			printf("Error during BZIP compression of one of the blocks");
			break;
		case 5:
			printf("Error generating the output file in the specified location");
			break;
		default:
			printf("Error writing the image");
		}
	}

	return error;
}


//============================================================================
//declare all possible instantitation for the template
template class multiviewImage<uint16_t>;
template class multiviewImage<uint8_t>;
template class multiviewImage<float>;