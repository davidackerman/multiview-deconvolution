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

#include <stdio.h>
#include "multiviewImage.h"
#include "external/klb/src/klb_imageIO.h"
#include "external/klb/src/klb_Cwrapper.h"
#include "cuda.h"
#include "book.h"
#include "imgUtils.h"
#include "external/imwarp_flexible/imwarp_flexible.h"


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
	clear();
};

//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::clear()
{
	for (size_t ii = 0; ii < imgVec_CPU.size(); ii++)
	{
		if (imgVec_CPU[ii] != NULL)
		{
			delete[]imgVec_CPU[ii];
			imgVec_CPU[ii] = NULL;
		}

		if (imgVec_GPU[ii] != NULL)
		{
			HANDLE_ERROR(cudaFree(imgVec_GPU[ii]));
			imgVec_GPU[ii] = NULL;
		}
	}

	imgVec_CPU.clear();
	imgVec_GPU.clear();
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
		imgVec_GPU[pos] = NULL;
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
void multiviewImage<imgType>::copyView_extPtr_to_CPU(size_t pos, const imgType* ptr, std::int64_t dims[MAX_DATA_DIMS])
{
	if (pos >= imgVec_CPU.size())
		return;

	if (pos >= dimsImgVec.size())
		dimsImgVec.resize(pos + 1);

	int64_t imSize = 1;
	dimsImgVec[pos].ndims = MAX_DATA_DIMS;
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
	{
		imSize *= dims[ii];
		dimsImgVec[pos].dims[ii] = dims[ii];
	}
	
	if (imgVec_CPU[pos] != NULL)
		delete[] imgVec_CPU[pos];

	imgVec_CPU[pos] = new imgType[imSize];
	memcpy(imgVec_CPU[pos], ptr, sizeof(imgType)* imSize);
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
std::int64_t multiviewImage<imgType>::numBytes(size_t pos) const
{	
	return (numElements(pos) * sizeof(imgType));
};

//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::copyROI(const imgType *p, std::int64_t dims[MAX_DATA_DIMS], int ndims, int pos, const klb_ROI& ROI)
{	
	if (ndims != 3)
	{
		cout << "ERROR: TODO:multiviewImage<imgType>::copyROI: code is only ready for 3 dimensions" << endl;
		exit(3);
	}

	imgType* imgA = new imgType[ROI.getSizePixels()];

	//copy elements
	int64_t offsetROI = 0;
	int64_t offsetFullImg;
	int64_t strideROI = ROI.getSizePixels(0); 	
	for (int64_t zz = ROI.xyzctLB[2]; zz <= ROI.xyzctUB[2]; zz++)//UB is inlcuded in ROI
	{
		offsetFullImg = ((int64_t)(ROI.xyzctLB[0])) + dims[0] * (((int64_t)(ROI.xyzctLB[1])) + dims[1] * zz);
		for (int64_t yy = ROI.xyzctLB[1]; yy <= ROI.xyzctUB[1]; yy++)//UB is inlcuded in ROI
		{
			//offsetFullImg = ROI.xyzctLB[0] + dims[0] * (yy + dims[1] * zz);

			memcpy(&(imgA[offsetROI]), &(p[offsetFullImg]), sizeof(imgType)* strideROI);
			offsetROI += strideROI;
			offsetFullImg += dims[0];
		}
	}


	if (pos < 0)
	{
		imgVec_CPU.push_back(imgA);
		imgVec_GPU.push_back(NULL);
		dimsImgVec.push_back(dimsImg());
		pos = imgVec_GPU.size() - 1;
	}
	else if (pos >= imgVec_CPU.size()){
		cout << "ERROR: multiviewImage<imgType>::copyROI: trying to place image in a view that does not exist" << endl;
		delete[] imgA;
		exit(3);
	}
	else{
		if (imgVec_CPU[pos] != NULL)
			delete[](imgVec_CPU[pos]);//TODO: if it was the same size, we do not need to reallocate

		imgVec_CPU[pos] = imgA;
	}


	//aupdate dimensions if necessary
	int ndims_ = 0;
	while (ndims_ < KLB_DATA_DIMS && ROI.getSizePixels(ndims_) > 1)
	{
		dimsImgVec[pos].dims[ndims_] = ROI.getSizePixels(ndims_);
		ndims_++;
	}
	dimsImgVec[pos].ndims = ndims_;


}

//=================================================================================
template<typename imgType>
KLB_DATA_TYPE klbDataTypeFromImgType()
{
    KLB_DATA_TYPE result ;
    if (is_integral<imgType>())
    {
        if (is_signed<imgType>())
        {
            switch (sizeof(imgType))
            {
                case 1:
                    result = KLB_DATA_TYPE::INT8_TYPE ;
                    break ;
                case 2:
                    result = KLB_DATA_TYPE::INT16_TYPE ;
                    break ;
                case 4:
                    result = KLB_DATA_TYPE::INT32_TYPE ;
                    break ;
                case 8:
                    result = KLB_DATA_TYPE::INT64_TYPE ;
                    break ;
                default:
                    // If get here, not a supported size.
                    throw std::logic_error("Internal error: Bad imgType in klbDataTypeFromImgType<imgType>()") ;
            }
        }
        else
        {
            // If get here, is unsigned
            switch (sizeof(imgType))
            {
                case 1:
                    result = KLB_DATA_TYPE::UINT8_TYPE ;
                    break ;
                case 2:
                    result = KLB_DATA_TYPE::UINT16_TYPE ;
                    break ;
                case 4:
                    result = KLB_DATA_TYPE::UINT32_TYPE ;
                    break ;
                case 8:
                    result = KLB_DATA_TYPE::UINT64_TYPE ;
                    break ;
                default:
                    // If get here, not a supported size.
                    throw std::logic_error("Internal error: Bad imgType in klbDataTypeFromImgType<imgType>()") ;
            }
        }
    }
    else
    {
        // If get here, is non-integral
        if (is_floating_point<imgType>())
        {
            switch (sizeof(imgType))
            {
                case 4:
                    result = KLB_DATA_TYPE::FLOAT32_TYPE ;
                    break ;
                case 8:
                    result = KLB_DATA_TYPE::FLOAT64_TYPE ;
                    break ;
                default:
                    // If get here, not a supported size.
                    throw std::logic_error("Internal error: Bad imgType in klbDataTypeFromImgType<imgType>()") ;
            }
        }
        else
        {
            // If get here, neither integral nor floating-point.  
            throw std::logic_error("Internal error: Bad imgType in klbDataTypeFromImgType<imgType>()") ;
        }
    }
    return result ;
} 

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
	
    // If imgType is float, and the file data type is uint8 or uint16, we convert on-the-fly.
    // Otherwise, the imgType must match the file data type.
    // Check now to avoid segfault later.
    KLB_DATA_TYPE imgTypeAsKLBDataType = klbDataTypeFromImgType<imgType>() ;
    KLB_DATA_TYPE fileDataType = imgFull.header.dataType ;
    bool areConvertingIntToFloat = (imgTypeAsKLBDataType == KLB_DATA_TYPE::FLOAT32_TYPE) && (fileDataType == KLB_DATA_TYPE::UINT8_TYPE || fileDataType == KLB_DATA_TYPE::UINT16_TYPE) ;
    if (!areConvertingIntToFloat)
    {
        if (imgTypeAsKLBDataType != fileDataType)
        {
            string imgDataTypeAsString = stringFromKLBDataType(imgTypeAsKLBDataType) ;
            string fileDataTypeAsString = stringFromKLBDataType(fileDataType) ;
            cout << "ERROR: multiviewImage<imgType>::readROI: image type (" << imgDataTypeAsString << ") does not match file image type (" << fileDataTypeAsString << "), and conversion is not supported" << endl;
            return 10;
        }
    }

    // Allocate storage for the image/ROI about to be read
	imgType* imgA = new imgType[ROI.getSizePixels()];		

    // Read in the data, converting if necessary
	if (areConvertingIntToFloat)
	{
        #ifdef _DEBUG
		    cout << "WARNING: multiviewImage<imgType>::readROI: converting int image to float32" << endl;
        #endif
		void* imgAint = malloc(ROI.getSizePixels() * imgFull.header.getBytesPerPixel());

		err = imgFull.readImage((char*)imgAint, &ROI, -1);
		if (err > 0)
			return err;
		
        switch (fileDataType)
        {
            case KLB_DATA_TYPE::UINT8_TYPE:
            {
                uint8_t* ptr = (uint8_t*)(imgAint);
                for (size_t ii = 0; ii < ROI.getSizePixels(); ii++)
                    imgA[ii] = (imgType)(ptr[ii]);
                break;
            }
            case KLB_DATA_TYPE::UINT16_TYPE:
            {
                uint16_t* ptr = (uint16_t*)(imgAint);
                for (size_t ii = 0; ii < ROI.getSizePixels(); ii++)
                    imgA[ii] = (imgType)(ptr[ii]);
                break;
            }
            default:
                string imgDataTypeAsString = stringFromKLBDataType(imgTypeAsKLBDataType) ;
                string fileDataTypeAsString = stringFromKLBDataType(fileDataType) ;
                cout << "ERROR: multiviewImage<imgType>::readROI: Unable to convert " << fileDataTypeAsString<< " to " << imgDataTypeAsString << endl;
                free(imgAint);
                delete[] imgA;
                return 20;
        }
		

		free(imgAint);
	}
	else
    {
        // Usual case with no conversion, so read normally
		err = imgFull.readImage((char*)imgA, &ROI, -1);
		if (err > 0)
			return err;
	}

    // Add imgA to self, dealing with memory management issues as needed
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
	else
    {		
		if (imgVec_CPU[pos] != NULL)
			delete[] (imgVec_CPU[pos]);  //TODO: if it was the same size, we do not need to reallocate

		imgVec_CPU[pos] = imgA;		
	}

	// Update dimensions if necessary
	int ndims = 0;
	while (ndims < KLB_DATA_DIMS && ROI.getSizePixels(ndims) > 1)
	{
		dimsImgVec[pos].dims[ndims] = ROI.getSizePixels(ndims);
		ndims++;
	}
	dimsImgVec[pos].ndims = ndims;

    // Return
	return err;
}

//===========================================================================================
template<class imgType>
void multiviewImage<imgType>::padArrayWithZeros(size_t pos, const std::uint32_t *dimsAfterPad)
{ 
	imgType *aux = fa_padArrayWithZeros(imgVec_CPU[pos],dimsImgVec[pos].dims, dimsAfterPad, dimsImgVec[pos].ndims); 

    //swap
	delete[] (imgVec_CPU[pos]);
	imgVec_CPU[pos] = aux;

    //reset dimensions
	for (int ii = 0; ii < dimsImgVec[pos].ndims; ii++)
		dimsImgVec[pos].dims[ii] = dimsAfterPad[ii];
};

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
int multiviewImage<imgType>::readImageSizeFromHeader(const std::string& filename, int64_t dimsOut[MAX_DATA_DIMS])
{

	klb_imageIO imgFull(filename);
	int err = 0;

	err = imgFull.readHeader();
	if (err > 0)
		return err;

	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		dimsOut[ii] = imgFull.header.xyzct[ii];

	return 0;
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

//===========================================================================================
template<class imgType>
int multiviewImage<imgType>::writeImage_uint16(const std::string& filename, int pos, float scale)
{
	if (pos >= imgVec_CPU.size() || imgVec_CPU[pos] == NULL)
		return 0;

	//initialize I/O object	
	klb_imageIO imgIO(filename);

	uint32_t xyzct[KLB_DATA_DIMS];
	uint64_t imSize = 1;
	for (int ii = 0; ii < dimsImgVec[pos].ndims; ii++)
	{
		xyzct[ii] = dimsImgVec[pos].dims[ii];
		imSize *= dimsImgVec[pos].dims[ii];
	}
	for (int ii = dimsImgVec[pos].ndims; ii <KLB_DATA_DIMS; ii++)
	{
		xyzct[ii] = 1;
	}

	//set header
	int error;
	switch (sizeof(imgType))//TODO: this is not accurate since int8 will be written as uint8
	{
	case 1:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT8_TYPE);
		//write image
		error = imgIO.writeImage((char*)(imgVec_CPU[pos]), -1);//all the threads available
		break;
	case 2:
		imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT16_TYPE);
		//write image
		error = imgIO.writeImage((char*)(imgVec_CPU[pos]), -1);//all the threads available
		break;
	case 4:
	{
			  imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT16_TYPE);
			  //normalize image
			  float Imax = -numeric_limits<float>::max();
			  float Imin = -Imax;
			  for (uint64_t ii = 0; ii < imSize; ii++)
			  {
				  Imax = max(Imax, (float)(imgVec_CPU[pos][ii]));
				  Imin = min(Imin, (float)(imgVec_CPU[pos][ii]));
			  }
			  
			  uint16_t* imUint16 = new uint16_t[imSize];
			  for (uint64_t ii = 0; ii < imSize; ii++)
			  {
				  imUint16[ii] = (uint16_t)(scale * (imgVec_CPU[pos][ii] - Imin) / (Imax - Imin));
			  }

			  //save scaling factor in header's metadata
			  sprintf(imgIO.header.metadata, "scale=%.8f;Imin=%.8f;Imax=%.8f", scale, Imin, Imax);

			  //write image
			  error = imgIO.writeImage((char*)(imUint16), -1);//all the threads available
			  delete[] imUint16;
			  break;
	}
	default:
		cout << "ERROR: format not supported yet" << endl;
		return 10;
	}


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



//===========================================================================================
template<class imgType>
int multiviewImage<imgType>::writeImageRaw(const std::string& filename, int pos)
{
	if (pos >= imgVec_CPU.size() || imgVec_CPU[pos] == NULL)
		return 0;

	FILE* fid = fopen(filename.c_str(), "wb");

	if (fid == NULL)
	{
		printf("Error opening file %s to save raw image data\n", filename.c_str());
		return 2;
	}

	fwrite(imgVec_CPU[pos],sizeof(imgType), numElements(pos), fid);
	fclose(fid);

	//write header information
	string filenameH(filename + ".txt");
	fid = fopen(filenameH.c_str(), "w");
	if (fid == NULL)
	{
		printf("Error opening file %s to save header\n", filenameH.c_str());
		return 2;
	}

	for (int ii = 0; ii < dimsImgVec[pos].ndims; ii++)
	{
		fprintf(fid,"%d ",(int)(dimsImgVec[pos].dims[ii]));
	}
	fprintf(fid, "\n");

	switch (sizeof(imgType))
	{
	case 1:
		fprintf(fid, "uint8\n");
		break;
	case 2:
		fprintf(fid, "uint16\n");
		break;
	case 4:
		fprintf(fid, "single\n");
		break;
	default:
		fprintf(fid, "unkown\n");
		break;
	}
	

	fclose(fid);
	return 0;
}

//=========================================================================
template<class imgType>
void multiviewImage<imgType>::apply_affine_transformation_img(int pos, std::int64_t dimsOut[MAX_DATA_DIMS], float A[AFFINE_3D_MATRIX_SIZE], int interpMode)
{
	cout << "ERROR: TODO: multiviewImage<imgType>::apply_affine_transformation_img: not implemented for types other than float" << endl;
	exit(3);
}
//=========================================================================
template<>
void multiviewImage<float>::apply_affine_transformation_img(int pos, std::int64_t dimsOut[MAX_DATA_DIMS], float A[AFFINE_3D_MATRIX_SIZE], int interpMode)
{	
    // If pos is out-of-range, return
	if (imgVec_CPU.size() < pos)
		return;

    // Determine the number of elements needed in the output array imOut
	int64_t imOutSize = 1;
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		imOutSize *= dimsOut[ii];

	//allocate memory for transformed image
	float* imOut = new float[imOutSize];

	//perform transformation
	imwarpFast_MatlabEquivalent(getPointer_CPU(pos), imOut, dimsImgVec[pos].dims, dimsOut, A, interpMode);

	//update image dimensions and pointer
	deallocateView_CPU(pos);
	imgVec_CPU[pos] = imOut;
	for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
		dimsImgVec[pos].dims[ii] = dimsOut[ii];
}

//=========================================================================
template<class imgType>
void multiviewImage<imgType>::apply_affine_transformation_psf(int pos, double A[AFFINE_3D_MATRIX_SIZE], int interpMode)
    {
    cout << "ERROR: TODO: multiviewImage<imgType>::apply_affine_transformation_psf: not implemented for types other than float" << endl;
    exit(3);
    }

//=========================================================================
template<>
void multiviewImage<float>::apply_affine_transformation_psf(int pos, double A[AFFINE_3D_MATRIX_SIZE], int interpMode)
    {
    // If pos is out-of-range, return
    if (imgVec_CPU.size() < pos)
        return;

    // Get the dims of the raw PSF
    int64_t rawPSFDims[3] ;
    rawPSFDims[0] = dimsImgVec[pos].dims[0] ;
    rawPSFDims[1] = dimsImgVec[pos].dims[1] ;
    rawPSFDims[2] = dimsImgVec[pos].dims[2] ;

    int64_t j_nonzero = find_first_nonzero_element_3d(getPointer_CPU(pos), rawPSFDims) ;  // this is just for debugging

    double rawPSFOrigin[3] = { 0.5, 0.5, 0.5 } ;  // Irrelevant, but let's be consistent with Matlab imwarp() defaults.
    double rawPSFSpacing[3] = { 1.0, 1.0, 1.0 } ;  // Generally not really true, but again, let's be consistent with Matlab imwarp() defaults
    double psfSpacing[3] = { 1.0, 1.0, 1.0 } ;
    int64_t psfDims[3] ;
    double psfOrigin[3] ;
    transform_lattice_3d(psfDims, psfOrigin,
                         A,
                         rawPSFDims, rawPSFOrigin, rawPSFSpacing,
                         psfSpacing) ;

    cout << "Origin of transformed PSF (xyz):  " << psfOrigin[0] << "  " << psfOrigin[1] << "  " << psfOrigin[2] << endl ;

    // Determine the number of elements needed in the output array psf
    int64_t psfElementCount = element_count_from_dims_3d(psfDims) ;

    //allocate memory for transformed image
    float* psf = new float[psfElementCount] ;

    //// Make float versions of origins and spacing, and transform
    //float AAsFloat[16] ;
    //float_from_double(AAsFloat, A, 16) ;
    //float rawPSFOriginAsFloat[3] ;
    //float_from_double(rawPSFOriginAsFloat, rawPSFOrigin, 3) ;
    //float rawPSFSpacingAsFloat[3] ;
    //float_from_double(rawPSFSpacingAsFloat, rawPSFSpacing, 3) ;
    //float psfOriginAsFloat[3] ;
    //float_from_double(psfOriginAsFloat, psfOrigin, 3) ;
    //float psfSpacingAsFloat[3] ;
    //float_from_double(psfSpacingAsFloat, psfSpacing, 3) ;

    //perform transformation
    //imwarpFast_MatlabEquivalent(getPointer_CPU(pos), psf, rawPSFDims, psfDims, A, interpMode) ;
    float* rawPSF = getPointer_CPU(pos) ;
    bool is_background_black = (interpMode & 1) ;  // bit 0 indicates whether background should be treated as black or extrapolated
    bool is_cubic = (interpMode & 2) ; // bit 1 indicates whether cubic interpolation should be used (as opposed to linear)
    imwarp_flexible(rawPSF, rawPSFDims, rawPSFOrigin, rawPSFSpacing,
                    psf, psfDims, psfOrigin, psfSpacing,
                    A,
                    is_cubic, is_background_black) ;

    // Print the size of the transformed PSF
    cout << "Size of transformed (but not trimmed) PSF:  " << psfDims[0] << "  " << psfDims[1] << "  " << psfDims[2] << endl ;

    // Output the transformed but-not-yet trimmed image 
    uint32_t output_yxzct[KLB_DATA_DIMS] = { (uint32_t)psfDims[0], (uint32_t)psfDims[1], (uint32_t)psfDims[2], 1, 1 } ;
    char output_stack_file_name[256] ;
    sprintf(output_stack_file_name, "untrimmed_PSF_%d.klb", pos) ;
    writeKLBstack(psf, output_stack_file_name, output_yxzct, KLB_DATA_TYPE::FLOAT32_TYPE, -1, NULL, NULL, KLB_COMPRESSION_TYPE::BZIP2, NULL);

    int64_t i_nonzero = find_first_nonzero_element_3d(psf, psfDims) ;  // this is just for debugging

    // Trim near-zero elements on all sides, keeping the PSF centered
    int64_t trimmedPSFDims[MAX_DATA_DIMS] ;
    float* trimmedPSF = trim_psf_3d(trimmedPSFDims, psf, psfDims) ;  // trimmedPSF allocated on heap via new operator.
     
    // Update image dimensions and pointer
    deallocateView_CPU(pos);
    imgVec_CPU[pos] = trimmedPSF;
    for (int ii = 0; ii < MAX_DATA_DIMS; ii++)
        dimsImgVec[pos].dims[ii] = trimmedPSFDims[ii];

    // Delete the untrimmed PSF
    delete [] psf ;
    }

//=========================================================================
template<class imgType>
void multiviewImage<imgType>::subtractBackground(size_t pos, imgType imgBackground)
{	
	imgType* ptr = getPointer_CPU(pos);
	if (ptr == NULL)
		return;

	for (int64_t ii = 0; ii < numElements(pos); ii++)
	{
		if (ptr[ii] > imgBackground)
			ptr[ii] -= imgBackground;
		else
			ptr[ii] = 0;
	}
}

//============================================================================
//declare all possible instantitation for the template
template class multiviewImage<uint16_t>;
template class multiviewImage<uint8_t>;
template class multiviewImage<float>;
