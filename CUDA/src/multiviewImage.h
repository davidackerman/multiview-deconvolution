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
* \brief holds a set of pointers to image files.
*/

#ifndef __MULTI_VIEW_IMAGE_HEADER_H__
#define __MULTI_VIEW_IMAGE_HEADER_H__


#define MAX_DATA_DIMS 3 //xyzct


#include <assert.h>
#include "external/imWarpFast/affine_transform_3d_single.h"
#include <vector>
#include <string>
#include <cstdint>
//#include <cstring>


//forward declaration
class klb_ROI;

struct dimsImg
{	
	std::int64_t dims[MAX_DATA_DIMS];
	int ndims;//in case we have fewer dimensions than expected (for example, 2D)

	dimsImg()
	{
		ndims = 0;
	}
	dimsImg(const dimsImg& p) : ndims(p.ndims)
	{
		memcpy(dims, p.dims, sizeof(std::int64_t) * MAX_DATA_DIMS);
	}
	dimsImg& operator=(const dimsImg& p)
	{
		if (this != &p)
		{
			ndims = p.ndims;
			memcpy(dims, p.dims, sizeof(std::int64_t) * MAX_DATA_DIMS);
		}
		return *this;
	}
};

template<class imgType>
class multiviewImage
{

public:

	//public variables
	std::vector<dimsImg> dimsImgVec;

	//constructor/desctructor
	multiviewImage();
	multiviewImage(size_t numViews);
	~multiviewImage();
	void clear();//release all memory

	//I/O functions
	int readImage(const std::string& filename, int pos);//if pos<0 then we add one image to the vector
	static int readImageSizeFromHeader(const std::string& filename, int64_t dimsOut[MAX_DATA_DIMS]);
	int readROI(const std::string& filename, int pos, const klb_ROI& ROI);
	void copyROI(const imgType *p, std::int64_t dims[MAX_DATA_DIMS], int ndims, int pos, const klb_ROI& ROI);//copy ROI form another image in memory
	int writeImage(const std::string& filename, int pos);	
	int writeImage_uint16(const std::string& filename, int pos, float scale);
	int writeImageRaw(const std::string& filename, int pos);
	static std::string recoverFilenamePatternFromString(const std::string& imgPath, int frame);
	static int getImageDimensionsFromHeader(const std::string& filename, std::uint32_t xyzct[MAX_DATA_DIMS]);

	void apply_affine_transformation_img(int pos, std::int64_t dimsOut[MAX_DATA_DIMS], float A[AFFINE_3D_MATRIX_SIZE], int interpMode);
	void subtractBackground(size_t pos, imgType imgBackground);

	//short IO functions
	size_t getNumberOfViews() const{ return imgVec_CPU.size(); };
	void resize(size_t numViews) { imgVec_CPU.resize(numViews, NULL); imgVec_GPU.resize(numViews, NULL); dimsImgVec.resize(numViews); };
	imgType* getPointer_CPU(size_t pos) { return(imgVec_CPU.size() <= pos ? NULL : imgVec_CPU[pos]);};
	imgType* getPointer_CPU(size_t pos) const { return(imgVec_CPU.size() <= pos ? NULL : imgVec_CPU[pos]); };
	void setPointer_CPU(size_t pos, imgType* ptr) { assert(imgVec_CPU.size() > pos); assert(imgVec_CPU[pos] == NULL); imgVec_CPU[pos] = ptr; };//use this function with a lot of caution (easy to have memory leaks
	imgType* getPointer_GPU(size_t pos) { return(imgVec_GPU.size() <= pos ? NULL : imgVec_GPU[pos]); };
	imgType* getPointer_GPU(size_t pos) const { return(imgVec_GPU.size() <= pos ? NULL : imgVec_GPU[pos]); };
	std::int64_t numElements(size_t pos) const;
	std::int64_t numBytes(size_t pos) const;
	void deallocateView_CPU(size_t pos);
	void deallocateView_GPU(size_t pos);
	void allocateView_GPU(size_t pos, size_t numBytes);
	void allocateView_CPU(size_t pos, size_t numElements); //this function does not setup dimsImgVec
	void copyView_GPU_to_CPU(size_t pos);
	void copyView_CPU_to_GPU(size_t pos);
	void copyView_extPtr_to_CPU(size_t pos, const imgType* ptr, std::int64_t dims[MAX_DATA_DIMS]);
	void setImgDims(size_t pos, const dimsImg &d);

	void padArrayWithZeros(size_t pos, const std::uint32_t *dimsAfterPad);

	
protected:

private:

	std::vector<imgType*> imgVec_CPU;
	std::vector<imgType*> imgVec_GPU;
	
};

#endif 
