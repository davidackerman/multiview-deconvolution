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
* \brief main interface to execute multiview deconvolution 
*/

#ifndef __MULTI_VIEW_DECONVOLUTION_IMAGE_HEADER_H__
#define __MULTI_VIEW_DECONVOLUTION_IMAGE_HEADER_H__

#include <string>
#include "multiviewImage.h"


typedef float weightType;
typedef float psfType;
typedef float outputType;

//from cufft.h to avoid including a cuda header here
typedef int cufftHandle;

//forward declaration
class klb_ROI;

template<class imgType>
class multiviewDeconvolution
{

public:


	//main objects to hold values
	multiviewImage<weightType> weights;
	multiviewImage<psfType> psf;
	multiviewImage<imgType> img;

	//constructor / destructor
	multiviewDeconvolution();	
	~multiviewDeconvolution();

	
	int readImage(const std::string& filename, int pos, const std::string& type);
	int readROI(const std::string& filename, int pos, const std::string& type, const klb_ROI& ROI);
	int writeDeconvoutionResult(const std::string& filename){ return J.writeImage(filename, 0); };
	int writeDeconvoutionResultRaw(const std::string& filename){ return J.writeImageRaw(filename, 0); };

	//perfoms all the preallocation and precalculation for the deconvolution
	int allocate_workspace(imgType imgBackground);
	int allocate_workspace_init_multiGPU(const uint32_t blockDims[MAX_DATA_DIMS], bool useWeights);
	int allocate_workspace_update_multiGPU(imgType imgBackground, bool useWeights);
	
	//different deconvolution methods
	void deconvolution_LR_TV(int numIters, float lambdaTV);//lucy-richardson with totalvariation regularization

	//regularization methods
	void regularization_TV(outputType* Jreg, outputType** temp_CUDA, int tempN );//temp_CUDA are arrays that are already preallocated and we can re-use. They are the same size as the output image

	//set/get IO functions
	void setNumberOfViews(int numViews);
	void copyDeconvoutionResultToCPU(){ J.copyView_GPU_to_CPU(0); };
	std::int64_t numElements_img(size_t pos){ return img.numElements(pos); };
	void padArrayWithConstant(const std::uint32_t *dimsAfterPad, int pos, const std::string& type,imgType constant);
	outputType* getJpointer() const { return J.getPointer_CPU(0); };

	//straight deconvolution from beginning to end
	static imgType* convolution3DfftCUDA(const imgType* im, const std::int64_t* imDim, const imgType* kernel, const std::int64_t* kernelDim, int devCUDA);
	imgType* convolution3DfftCUDA_img_psf(size_t pos, int devCUDA);
	void calculateWeights(size_t pos, float anisotropyZ);

	//debuggin methods
	void debug_writDeconvolutionResultRaw(const std::string& filename);
	static void debug_writeGPUarray(float* ptr_GPU, dimsImg& dims, const std::string& filename);
	static void debug_writeCPUarray(float* ptr_CPU, dimsImg& dims, const std::string& filename);
    void debug_writeCPUarray_img(size_t pos, const std::string& filename){ debug_writeCPUarray(img.getPointer_CPU(pos), img.dimsImgVec[pos], filename); };
	void debug_writeGPUarray_weights(size_t pos, const std::string& filename){ debug_writeGPUarray(weights.getPointer_GPU(pos), img.dimsImgVec[pos], filename); };
	static outputType* debug_regularization_TV_CPU(const outputType* f, const std::int64_t* imDim);
	outputType* debug_regularization_TV_GPU(const outputType* f, const std::int64_t* imDim);
	

protected:	
	

	multiviewImage<outputType> J;//holds the output, so it is going to have a different number of views

	//holds the fft plans 
	cufftHandle fftPlanFwd, fftPlanInv;

private:
};

#endif 