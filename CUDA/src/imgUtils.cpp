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
int copySliceOld(float* target, int64_t* targetSize,
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
        int64_t elementsPerTargetSlice = elementsPerSliceOld(arity, targetSize, sliceArity) ;
        int64_t elementsPerSourceSlice = elementsPerSliceOld(arity, sourceSize, sliceArity) ;
        int64_t nElementsToTrimInThisDimension = nElementsToTrim[iDim] ;
        float* sourcePlusOffset = source + elementsPerSourceSlice*nElementsToTrimInThisDimension ;
        for (int64_t i = 0; i < targetSize[iDim]; ++i)
            {
            float* targetSlice = target + elementsPerTargetSlice*i ;
            float* sourceSlice = sourcePlusOffset + elementsPerSourceSlice*i ;
            int err = copySliceOld(targetSlice, targetSize + 1, nElementsToTrim + 1, arity - 1, sliceArity - 1, sourceSlice, sourceSize + 1) ;
            if (err) { return err ; }
            }
        return 0 ;
        }
    }

//=========================================================================
int copySlice(float* target, int64_t* targetSize,
              int64_t* nElementsToTrim,
              int64_t arity, 
              float* source, int64_t* sourceSize)
{
    // Copy the source array to the target array, trimming the given number of elements off each end of each dimension of the source array.
    // Arity is the dimensionality of both target and source. targetSize, sourceSize, and nElementsToTrim should all be of length arity.
    // sliceArity gives the number of dimensions remaining to be copied over.
    // Returns 0 if everything works, a negative error code otherwise.
    int64_t iDim = arity - 1 ;  // The dimension we will work on, always the last one
    if (arity == 1)
    {
        // If only on dimension left, a simple memcpy() should do it
        memcpy(target, &(source[nElementsToTrim[iDim]]), sizeof(float)*targetSize[iDim]) ;
        return 0 ;
    }
    else
    {
        int64_t elementsPerTargetSlice = elementsPerSlice(arity, targetSize) ;
        int64_t elementsPerSourceSlice = elementsPerSlice(arity, sourceSize) ;
        int64_t nElementsToTrimInThisDimension = nElementsToTrim[iDim] ;
        float* sourcePlusOffset = source + elementsPerSourceSlice*nElementsToTrimInThisDimension ;
        for (int64_t iTargetSlice = 0; iTargetSlice < targetSize[iDim]; ++iTargetSlice)
        {
            float* targetSlice = target + elementsPerTargetSlice*iTargetSlice ;
            float* sourceSlice = sourcePlusOffset + elementsPerSourceSlice*iTargetSlice ;
            int err = copySlice(targetSlice, targetSize, nElementsToTrim, arity - 1, sourceSlice, sourceSize) ;
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
int64_t elementsPerSliceOld(int64_t arity, int64_t* size, int64_t sliceArity)
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

//=========================================================================
int64_t elementsPerSlice(int64_t arity, int64_t* dims)
{
    // Computes the number of elements in a slice of an array.  We assume the 
    // slice corresponds to the first arity-1 dimensions of the array.
    // dims is of length arity, and gives the dimensions of the array in question.
    // Basically, this is the product of all elements of dims except for the last one.

    int64_t result = 1 ;
    for (int64_t iDim = 0; iDim < arity-1; iDim++)
    {
        result *= dims[iDim] ;
    }
    return result ;
}

//=========================================================================
void transform_lattice_3d(int64_t* targetDims, double* targetOrigin, 
                          float* A,
                          int64_t* sourceDims, double* sourceOrigin, double* sourceSpacing,
                          double * targetSpacing)
    {
    // Given an affine transform and a source lattice, computes a target lattice with the same 
    // spacing as the source lattice, that is designed to just include the image of the source 
    // lattice after it is transformed by A.  A "lattice" is defined by an axis-aligned cuboid in 
    // space combined with a number of voxels in each dimension.  The corner of the cuboid closest to the
    // origin is the same point as the corner of the first voxel closest to the origin.  Similarly, the 
    // corner of the cuboid farthest from the origin is the same point as the corner of the last voxel 
    // farthest from the origin.  I.e. there is no voxel center at the corner of the cuboid.
    //
    // Inputs:
    //   A is the transform matrix, serialized into an 1D array.  It should of size
    //     4^2==16.  It is stored row-major, assuming y = A*x is used for the transform, with y and x col 
    //     vectors.  Or, equivalently, it can be viewed as being stored col-major, assuming y = x*A is 
    //     used for the transform, with y and x row vectors.  Taking the first view, the last row should 
    //     be 0 0 0 1.
    //   sourceDims is the size of the source lattice in each dimension.
    //   sourceOrigin is the location of the "lowermost" corner of the cuboid associated with the lattice.
    //   sourceSpacing is the spacing between voxels in each dimension.  
    //     The "uppermost" corner of the source cuboid is located at sourceOrigin+sourceSize.*sourceSpacing.
    //   targetSpacing is the desired spacing between voxels in the output lattice.
    //
    // Outputs:
    //   targetDims is the size of the target lattice in each dimension.
    //   targetOrigin is the location of the "lowermost" corner of the cuboid associated with the lattice.

    double sourceExtent[3] ;
    extent_from_dims_and_spacing_3d(sourceExtent, sourceDims, sourceSpacing) ;

    double idealTargetOrigin[3] ;
    double idealTargetExtent[3] ;
    transform_cuboid_3d(idealTargetOrigin, idealTargetExtent,
                        A,
                        sourceOrigin, sourceExtent) ;

    lattice_from_cuboid_3d(targetDims, targetOrigin, 
                           idealTargetOrigin, idealTargetExtent, targetSpacing) ;

    }

//=========================================================================
void transform_cuboid_3d(double* targetOrigin, double* targetExtent,
                         float* A,
                         double* sourceOrigin, double* sourceExtent)
    {
    // Given a cuboid in a source space, computes that cuboid that just 
    // contains the image of it in a target space, after the source cuboid goes through the affine transform defined by A.
    // targetOrigin and targetExtent are outputs, and should be preallocated to length 3 by the caller.
    // A should be of length 16, representing a 4x4 affine transform array.  It is stored row-major, assuming y = A*x is 
    // used for the transform, with y and x col vectors.  Or, equivalently, it can be viewed as being stored col-major, assuming 
    // y = x*A is used for the transform, with y and x row vectors.  Taking the first view, the last row should be 0 0 0 1.

    /*
    // Do the matrix multiply, treating the vectors as col vectors, and A as row-major
    double sourceOriginImage[3] ;
    double sourceExtentImage[3] ;
    affine_transform_3d(sourceOriginImage, A, sourceOrigin) ;
    affine_transform_3d(sourceExtentImage, A, sourceExtent) ;
    */

    // Generate all the corners of the image in the target space
    double sourceCorner[3] ;
    double targetCorners[8][3] ;  
    double keep[3] ;
    size_t n = 0 ;
    for (size_t i = 0; i < 2; ++i)
        {
        keep[0] = double(i) ;
        for (size_t j = 0; j < 2; ++j)
            {
            keep[1] = double(j) ;
            for (size_t k = 0; k < 2; ++k)
                {
                keep[2] = double(k) ;
                elementwise_product_3d(sourceCorner, keep, sourceExtent) ;
                sum_in_place_3d(sourceCorner, sourceOrigin) ;
                affine_transform_3d(targetCorners[n], A, sourceCorner) ;
                ++n ;
                }
            }
        }

    // Extract the max and min in each dimension to find the bounding box targetCorners
    const float inf = std::numeric_limits<float>::infinity() ;
    targetOrigin[0] = inf ;
    targetOrigin[1] = inf ;
    targetOrigin[2] = inf ;
    double targetOriginPlusExtent[3] = { -inf, -inf, -inf } ;
    for (size_t i = 0; i < 8; i++)
        for (size_t j = 0; j < 3; j++)
            {
            double targetCornerElement = targetCorners[i][j] ;
            if (targetCornerElement<targetOrigin[j])
                targetOrigin[j] = targetCornerElement ;
            // Can't do else-if here b/c both are true when targetOrigin, targetOriginPlusExtent have initial vals (inf and -inf, respectively)
            if (targetCornerElement>targetOriginPlusExtent[j])  
                targetOriginPlusExtent[j] = targetCornerElement ;
            }

    // Take the difference of the bounding box targetCorners to get the extent
    difference_3d(targetExtent, targetOriginPlusExtent, targetOrigin) ;
    }

//=========================================================================
void lattice_from_cuboid_3d(int64_t* dims, double* origin,
                            double* ideal_origin, double* ideal_extent, double *spacing)
    {
    // Given a bounding box and a spacing, compute a (generally) slightly larger bounding box
    // centered on the original, with extent an integer multiple of spacing (in each dimension).
    // The resulting bounding box, along with the integer multiple in each dimension, comprise a 
    // "lattice": a set of regularly spaced points in Cartesian space for which voxel data might 
    // be determined.

    double ideal_count[3] ;
    elementwise_quotient_3d(ideal_count, ideal_extent, spacing) ;

    dims_from_extent_with_unit_spacing_3d(dims, ideal_count) ;

    // From the dims and the spacing in each dimension, easy to get the final extent
    double extent[3] ;
    extent_from_dims_and_spacing_3d(extent, dims, spacing) ;
    
    double ideal_radius[3] ;
    scalar_product_3d(ideal_radius, 0.5, ideal_extent) ;

    double center[3] ;
    sum_3d(center, ideal_origin, ideal_radius) ;  // This is the center of both the ideal cuboid and the final one

    double radius[3] ;
    scalar_product_3d(radius, 0.5, extent) ;

    difference_3d(origin, center, radius) ;
    }

/*
 void row_times_matrix(double* y, double* x, float *A)
     {
     // We assume A represents a 4x4 matrix stored row-major
     for (size_t j = 0; j < 3; j++)
         {
         y[j] == 0.0 ;
         for (size_t i = 0; i < 3; i++)
             y[j] += x[i] * A[4 * i + j] ;
         y[j] += A[4 * 3 + j] ;
         }
     }
*/

void affine_transform_3d(double y[3], float T[16],  double x[3])
    {
    // We assume T represents a 4x4 affine transform matrix stored row-major.
    // I.e. T is of the form [ A b ]
    //                       [ 1 0 ] ,
    // where A is 3x3, and b is 3x1.
    // This does y = A*x + b , with y and x treated as col vectors.
    for (size_t i = 0; i < 3; i++)
        {
        double yi = 0.0 ;
        for (size_t j = 0; j < 3; j++)
            yi += double(T[4 * i + j]) * x[j] ;
        yi += double(T[4 * i + 3]) ;
        y[i] = yi ;
        }
    }

void extent_from_dims_and_spacing_3d(double* extent, int64_t* dims, double* spacing)
    {
    extent[0] = double(dims[1]) * spacing[0] ;  // dims are in order n_y, n_x, n_z
    extent[1] = double(dims[0]) * spacing[1] ;
    extent[2] = double(dims[2]) * spacing[2] ;
    }

void elementwise_product_3d(double* z, double* x, double* y)
    {
    for (size_t i = 0; i < 3; i++)
        z[i] = x[i] * y[i] ;
    }

void scalar_product_3d(double* y, double a, double* x)
    {
    for (size_t i = 0; i < 3; i++)
        y[i] = a * x[i] ;
    }

void elementwise_quotient_3d(double* z, double* x, double* y)
    {
    for (size_t i = 0; i < 3; i++)
        z[i] = x[i] / y[i] ;
    }

void dims_from_extent_with_unit_spacing_3d(int64_t* dims, double* extent)
    {
    // dims are in order n_y, n_x, n_z
    dims[0] = int64_t(ceil(extent[1])) ;
    dims[1] = int64_t(ceil(extent[0])) ;
    dims[2] = int64_t(ceil(extent[2])) ;
    }

void difference_3d(double* z, double* x, double* y)
    {
    for (size_t i = 0; i < 3; i++)
        z[i] = x[i] - y[i] ;
    }

void sum_3d(double* z, double* x, double* y)
    {
    for (size_t i = 0; i < 3; i++)
        z[i] = x[i] + y[i] ;
    }

void sum_in_place_3d(double* y, double* x)
    {
    for (size_t i = 0; i < 3; i++)
        y[i] += x[i] ;
    }

int64_t element_count_from_dims_3d(int64_t* dims)
    {
    int64_t element_count = 1 ;
    for (size_t i = 0; i < 3; i++)
        element_count *= dims[i] ;
    return element_count ;
    }

float normalize_in_place_3d(float* x, int64_t* dims)
    {
    float sum = 0.0f ;
    size_t n = size_t(element_count_from_dims_3d(dims)) ;
    for (size_t i = 0; i < n; ++i)
        sum += x[i] ;
    for (size_t i = 0; i < n; ++i)
        x[i] = x[i] / sum ;
    return sum ;  // why not?
    }

void pos_in_place_3d(float *x, int64_t dims[3])
    {
    // Determine the number of elements needed in the output array x
    int64_t n_elements = element_count_from_dims_3d(dims) ;

    // Eliminate any negative elements that might have resulted from cubic interpolation
    for (int64_t i = 0; i < n_elements; ++i)
        {
        if (x[i] < 0.0f)
            x[i] = 0.0f;
        }
    }

int64_t find_first_nonzero_element_3d(float *x, int64_t dims[3])
    {
    int64_t n_elements = element_count_from_dims_3d(dims) ;
    for (int64_t i=0; i<n_elements; ++i)
        {
        if (x[i]!=0.0f)
            return i ;
        }
    return -1 ; 
    }

void float_from_double(float* y, double* x, int64_t n)
    {
    for (int64_t i = 0; i < n; ++i)
        {
        y[i] = float(x[i]) ;
        }
    }

void determine_n_elements_to_trim_3d(int64_t* nElementsToTrim, float* psf, int64_t* psfDims, float threshold)
{
    // Figure out how many elements we're going to trim
    //float threshold = 1e-10f ;
    //int64_t nElementsToTrim[MAX_DATA_DIMS] ;
    for (int64_t iDim = 0; iDim < 3; iDim++)
    {
        int64_t nElementsThisDim = psfDims[iDim] ;
        int64_t indexOfFirstSuperthresholdSliceHere = indexOfFirstSuperthresholdSlice(psf, 3, psfDims, iDim, threshold) ;
        if (indexOfFirstSuperthresholdSliceHere < 0)
        {
            // This means all slices are empty
            nElementsToTrim[iDim] = 0 ;   // User is going to be sad later
        }
        else
        {
            int64_t putativeLowestIndexToKeep = indexOfFirstSuperthresholdSliceHere - 2 ;  // Add some padding
            int64_t lowestIndexToKeep = (putativeLowestIndexToKeep >= 0) ?
                                        putativeLowestIndexToKeep :
                                        0 ;  // Limit to allowable range
            int64_t nElementsToTrimAtLowEnd = lowestIndexToKeep ;  // Will be optimized away, never fear
            int64_t indexOfLastSuperthresholdSliceHere = indexOfLastSuperthresholdSlice(psf, 3, psfDims, iDim, threshold) ;  // This can't return -1, because indexOfFirstSuperthresholdSlice() would have already done so if all slices were empty
            int64_t putativeHighestIndexToKeep = indexOfLastSuperthresholdSliceHere + 2 ;  // Add some padding
            int64_t highestIndexToKeep = (putativeHighestIndexToKeep < nElementsThisDim) ?
                                         putativeHighestIndexToKeep :
                                         (nElementsThisDim - 1) ;   // Limit to allowable range
            int64_t nElementsToTrimAtHighEnd = (nElementsThisDim - 1) - highestIndexToKeep ;
            int64_t nElementsToTrimThis = (nElementsToTrimAtLowEnd < nElementsToTrimAtHighEnd) ? nElementsToTrimAtLowEnd : nElementsToTrimAtHighEnd ;  // Have to be symmetric, so pick the smaller
            /*
            if ( 2*nElementsToTrimThis>psfDims[iDim] )
            {
                std::cerr << "Houston, we have a problem" << endl ;
            }
            */
            nElementsToTrim[iDim] = nElementsToTrimThis ;

        }
    }
}


//=============================================================
template float* fa_padArrayWithZeros<float>(const float* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
template std::uint16_t* fa_padArrayWithZeros<std::uint16_t>(const std::uint16_t* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
template std::uint8_t* fa_padArrayWithZeros<std::uint8_t>(const std::uint8_t* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
