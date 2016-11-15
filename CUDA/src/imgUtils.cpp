/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  imgUtils.cpp
*
*  Created on : June 11th, 2015
* Author : Fernando Amat
*
* \brief random functions that are generically useful
*/

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include "imgUtils.h"


using namespace std;

template<class imgType>
imgType* fa_padArrayWithZeros(const imgType* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims)
{

	if (ndims != 3)
	{
		cout << "TODO:ERROR: padArrayWithZeros: function not ready for other values of ndims except for " << 3 << endl;
		exit(3);
	}

	int64_t nImg = 1;
	for (int ii = 0; ii < ndims; ii++)
	{
		if (dimsNow[ii] > dimsAfterPad[ii])
		{
			cout << "ERROR: padArrayWithZeros: new dimensions are smaller than current dimensions" << endl;
			return NULL;
		}
		nImg *= (int64_t)(dimsAfterPad[ii]);
	}



	imgType* p = new imgType[nImg];
	memset(p, 0, sizeof(imgType)* nImg);	

    //copy "lines" of x	
	size_t lineSize = dimsNow[0] * sizeof(imgType);
	int64_t idx = 0;
	int64_t count = 0;	
	for (int64_t zz = 0; zz < dimsNow[2]; zz++)
	{        
		idx = dimsAfterPad[0] * dimsAfterPad[1] * zz;		
		for (int64_t yy = 0; yy < dimsNow[1]; yy++)
		{
			//update for new array
			//idx = dimsAfterPad[0] * ( yy + dimsAfterPad[1] * zz);
			//update for new array
			//count = dimsNow[0] * (yy + dimsNow[1] * zz);

			//copy elements
			memcpy(&(p[idx]), &(im[count]), lineSize );
			
            //update counters
            idx += dimsAfterPad[0];
			count += dimsNow[0];
		}
	}

	return p;
}

//====================================================
std::string generateTempFilename(const char* prefix)
{
#ifdef _WIN32
	char s[MAX_PATH];
	if (GetTempPath(MAX_PATH, s)) {
		//std::cout << "GetTempPath() returned <" << s << ">\n";
	}
	else {
		std::cout << "ERROR: generateTempFilename: GetTempPath() failed with 0x" << std::hex << GetLastError() << "\n";
	}

	char* name = _tempnam("c:\\tmp", prefix);
	std::string nameS(name); 
	
#else //unix systems
	char *folder = getenv("TMPDIR");
	if (folder == NULL)
		folder = "/tmp";

	char *name = tempnam(folder, prefix);
	
	std::string nameS(name);		
#endif
		
	free(name);
	
	return nameS;

}

//=========================================================================
int copySlice(float* target, int64_t* targetSize,
              int64_t* nElementsToTrim,
              int64_t arity, int64_t sliceArity,
              float* source, int64_t* sourceSize)
    {
    // Copy the source array to the target array, trimming the given number of elements off each end of each dimension of the source array.
    // Arity is the dimensionality of both target and source. targetSize, sourceSize, and nElementsToTrim should all be of length arity.
    // sliceArity gives the number of dimensions remaining to be copied over.
    // Returns 0 if everything works, a negative error code otherwise.
    int64_t iDim = arity - sliceArity ;  // The dimension we will work on
    if (sliceArity == 1)
        {
        // If only on dimension left, a simple memcpy() should do it
        memcpy(target, &(source[nElementsToTrim[iDim]]), sizeof(float)*targetSize[iDim]) ;
        return 0 ;
        }
    else
        {
        int64_t elementsPerTargetSlice = elementsPerSlice(arity, targetSize, sliceArity) ;
        int64_t elementsPerSourceSlice = elementsPerSlice(arity, sourceSize, sliceArity) ;
        int64_t nElementsToTrimInThisDimension = nElementsToTrim[iDim] ;
        float* sourcePlusOffset = source + elementsPerSourceSlice*nElementsToTrimInThisDimension ;
        for (int64_t i = 0; i < targetSize[iDim]; ++i)
            {
            float* targetSlice = target + elementsPerTargetSlice*i ;
            float* sourceSlice = sourcePlusOffset + elementsPerSourceSlice*i ;
            int err = copySlice(targetSlice, targetSize + 1, nElementsToTrim + 1, arity - 1, sliceArity - 1, sourceSlice, sourceSize + 1) ;
            if (err) { return err ; }
            }
        return 0 ;
        }
    }

//=========================================================================
int64_t indexOfFirstSuperthresholdSlice(float *x, int64_t arity, int64_t* xDims, int64_t iDim, float threshold)
    {
    // Projects x down to a single 1D boolean vector of size xDims[iDim], which indicates 
    // if the subarray corresponding to that element is "superthreshold", i.e. contains any
    // superthreshold values.

    int64_t nElements = int64_t(xDims[iDim]) ;
    for (int64_t iElement = 0; iElement < nElements; ++iElement)
        {
        if (isSliceSuperthreshold(x, arity, xDims, iDim, iElement, threshold))
            return iElement ;
        }
    return int64_t(-1) ;
    }

//=========================================================================
int64_t indexOfLastSuperthresholdSlice(float *x, int64_t arity, int64_t* xDims, int64_t iDim, float threshold)
    {
    // Projects x down to a single 1D boolean vector of size xDims[iDim], which indicates 
    // if the subarray corresponding to that element is "superthreshold", i.e. contains any 
    // superthreshold values.

    int64_t nElements = int64_t(xDims[iDim]) ;
    for (int64_t iElement = nElements - 1; iElement >= 0; --iElement)
        {
        if (isSliceSuperthreshold(x, arity, xDims, iDim, iElement, threshold))
            return iElement ;
        }
    return int64_t(-1) ;
    }

//=========================================================================
bool isSliceSuperthreshold(float *x, int64_t arity, int64_t* xDims, int64_t iDim, int64_t iElement, float threshold)
    {
    // Returns whether the slice indicated by dimension iDim, element iElement is "superthreshold", 
    // meaning that any of its elements are superthreshold.

    int64_t sizeOfThisDim = int64_t(xDims[iDim]) ;
    int64_t nElementsPerChunk = chunkSize(xDims, iDim) ;
    int64_t nStride = sizeOfThisDim * nElementsPerChunk ;
    int64_t nChunks = chunkCount(arity, xDims, iDim) ;
    int64_t indexOfFirstElementInChunk = nElementsPerChunk * iElement ;
    for (int64_t chunkIndex = 0; chunkIndex < nChunks; ++chunkIndex)
        {
        int64_t indexOfElement = indexOfFirstElementInChunk ;
        for (int64_t indexOfElementWithinChunk = 0; indexOfElementWithinChunk < nElementsPerChunk; ++indexOfElementWithinChunk)
            {
            float value = x[indexOfElement] ;
            if (value > threshold)
                return true ;
            ++indexOfElement ;
            }
        indexOfFirstElementInChunk += nStride ;
        }
    // If we get here, we checked all the elements, and they were all subthreshold
    return false ;
    }

//=========================================================================
int64_t chunkSize(int64_t* xDims, int64_t iDim)
    {
    // Helper function for isSliceSuperthreshold().  The elements we need to examine occur 
    // in contiguous 'chunks' in the serialized array, and this returns the size of 
    // each chunk.

    int64_t result = 1 ;
    for (int64_t jDim = 0; jDim < int64_t(iDim); jDim++)
        {
        result *= xDims[jDim] ;
        }
    return result ;
    }

//=========================================================================
int64_t chunkCount(int64_t arity, int64_t* xDims, int64_t iDim)
    {
    // Helper function for isSliceSuperthreshold().  The elements we need to examine occur 
    // in contiguous 'chunks' in the serialized array, and this returns the number of 
    // such chunks.

    int64_t result = 1 ;
    for (int64_t jDim = int64_t(iDim) + 1; jDim < arity; jDim++)
        {
        result *= xDims[jDim] ;
        }
    return result ;
    }

//=========================================================================
int64_t elementsPerSlice(int64_t arity, int64_t* size, int64_t sliceArity)
    {
    // Computes the number of elements in a slice of an array.  We assume the 
    // slice corresponds to the last sliceArity dimensions of the array.
    // size is of length arity, and gives the dimensions of the array in question.
    // A sliceArity of 0 returns 1, and a sliceArity equal to arity returns the 
    // product of all elements of size.

    int64_t iDim = arity - sliceArity ;
    int64_t result = 1 ;
    for (int64_t jDim = iDim; jDim < arity; jDim++)
        {
        result *= size[jDim] ;
        }
    return result ;
    }

//=============================================================
template float* fa_padArrayWithZeros<float>(const float* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
template std::uint16_t* fa_padArrayWithZeros<std::uint16_t>(const std::uint16_t* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
template std::uint8_t* fa_padArrayWithZeros<std::uint8_t>(const std::uint8_t* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
