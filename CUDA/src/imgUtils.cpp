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
imgType* fa_padArrayWithConstant(const imgType* im,const std::int64_t *dimsNow,const std::uint32_t *dimsAfterPad,int ndims, imgType C) {

    if(ndims!=3) {
        cout<<"TODO:ERROR: padArrayWithZeros: function not ready for other values of ndims except for "<<3<<endl;
        exit(3);
    }

    int64_t nImg=1;
    for(int ii=0; ii < ndims; ii++) {
        if(dimsNow[ii] > dimsAfterPad[ii]) {
            cout<<"ERROR: padArrayWithZeros: new dimensions are smaller than current dimensions"<<endl;
            return NULL;
        }
        nImg*=(int64_t)(dimsAfterPad[ii]);
    }



    imgType* p=new imgType[nImg];
    for(int64_t i=0;i<nImg;++i)
        p[i]=C;

    //copy "lines" of x	
    size_t lineSize=dimsNow[0]*sizeof(imgType);
    int64_t idx=0;
    int64_t count=0;
    for(int64_t zz=0; zz < dimsNow[2]; zz++) {
        idx=dimsAfterPad[0]*dimsAfterPad[1]*zz;
        for(int64_t yy=0; yy < dimsNow[1]; yy++) {
            //update for new array
            //idx = dimsAfterPad[0] * ( yy + dimsAfterPad[1] * zz);
            //update for new array
            //count = dimsNow[0] * (yy + dimsNow[1] * zz);

            //copy elements
            memcpy(&(p[idx]),&(im[count]),lineSize);

            //update counters
            idx+=dimsAfterPad[0];
            count+=dimsNow[0];
        }
    }

    return p;
}

//template<class imgType>
//imgType* fa_padArrayWithZeros(const imgType* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims)
//{
//
//    return fa_padArrayWithConstant(im,dimsNow,dimsAfterPad,ndims,0);
//
//}

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
int copy_slice(float* target, int64_t* targetSize,
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
            int err = copy_slice(targetSlice, targetSize, nElementsToTrim, arity - 1, sourceSlice, sourceSize) ;
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
                          double* A,
                          int64_t* sourceDims, double* sourceOrigin, double* sourceSpacing,
                          double * targetSpacing)
    {
    // Given an affine transform and a source lattice, computes a target lattice with the same 
    // spacing as the source lattice, that is designed to just include the image of the source 
    // lattice after it is transformed by A.  A "lattice" is a set of regularly spaced points.  
    // Note that in the source lattive, the sample closest to the 
    // origin is at the point sourceOrigin + 0.5*sourceSpacing, *not* at sourceOrigin.  
    // Similarly for the target lattice.
    // 
    // This function is designed to compute the same target lattice as Matlab does when you call imwarp()
    // without specifying an explicit OutputView.
    //
    //
    // Inputs:
    //   A is the transform matrix, serialized into an 1D array.  It should of size
    //     4^2==16.  It is stored col-major, assuming y = x*A is used for the transform, with y and x row 
    //     vectors.  Thus last col should be 0 0 0 1, and the last row should hold the translation.
    //   sourceDims is the size of the source lattice in each dimension, in order yxz.  (*Not* xyz.)
    //   sourceOrigin gives the position of the source lattice in the source space.  
    //     In particular, the lattice point closest to the origin is at
    //     sourceOrigin + 0.5*sourceSpacing.  sourceOrigin is in order xyz.
    //   sourceSpacing is the spacing between samples in each dimension, in order xyz.
    //   targetSpacing is the desired spacing between voxels in the output lattice, in order xyz.
    //
    // Outputs:
    //   targetDims is the size of the target lattice in each dimension, in order yxz.  (*Not* xyz.)
    //   targetOrigin is the location of the target lattice in the target space.  In particular,
    //     the lattice point closest to the origin is at
    //     targetOrigin + 0.5*targetSpacing.  targetSpacing is in order xyz.

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
                         double* A,
                         double* sourceOrigin, double* sourceExtent)
    {
    // Given a cuboid in a source space, computes that cuboid that just 
    // contains the image of it in a target space, after the source cuboid goes through the affine transform defined by A.
    // targetOrigin and targetExtent are outputs, and should be preallocated to length 3 by the caller.
    // A should be of length 16, representing a 4x4 affine transform array.  It is stored col-major, assuming y = x*A is 
    // used for the transform, with y and x row vectors.  
    // targetOrigin, targetExtent, sourceOrigin, and sourceExtent are all three-vectors, in the order xyz.

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
                affine_transform_3d(targetCorners[n], sourceCorner, A) ;
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
    // be determined.  All args are arrays of length 3.  Dims is in order yxz, all others are in 
    // order xyz.

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

void affine_transform_3d(double y[3], double x[3], double T[16])
    {
    // We assume T represents a row-form 4x4 affine transform matrix stored col-major.
    // I.e. T is of the form [ A 1 ]
    //                       [ b 0 ] ,
    // where A is 3x3, and b is 1x3.
    // This does y = x*A + b , with y and x treated as row vectors.
    for (size_t i = 0; i < 3; i++)
        {
        double yi = 0.0 ;
        for (size_t j = 0; j < 3; j++)
            yi += T[4 * i + j] * x[j] ;
        yi += T[4 * i + 3] ;
        y[i] = yi ;
        }
    }

void extent_from_dims_and_spacing_3d(double* extent, int64_t* dims, double* spacing)
    {
    // Given 3D dimensions (in order yxz), and spacing between samples (in order xyz), 
    // computes the size of the cuboid covered by the implied lattice, in order xyz.
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
    // extent is in order x, y, z
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
    // Normalizes the given stack x so that on return its elements sum to unity.
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
    // Compute the elementwise pos() function on the elements of the stack x.  
    // pos(x) is equal to zero if x<0, and is otherwise equal to x.

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
    // Determines the serial index of the first nonzero element of the stack x,
    // or -1 if there is no such element.
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
    // Utility function used by trim_psf_3d() to determine how many elements to trim on each side, in each dimension.

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

float *trim_psf_3d(int64_t* trimmed_psf_dims, float* psf, int64_t* psf_dims)
{
    // Given an input stack, psf, returns a "substack" that includes all samples strictly greater than 1e-10f.
    // The substack is also centered on the original stack, i.e. the same number of slices are deleted 
    // on each side of the original stack.  
    // If all elements of the original stack are below threshold, simply returns the original stack.
    // 
    // Note that on return, the value *trimmedPSFPtr is allocated on the heap via new, and will need to be deleted by the caller

    // Figure out how many elements we're going to trim
    float threshold = 1e-10f ;
    int64_t n_elements_to_trim[3] ;
    determine_n_elements_to_trim_3d(n_elements_to_trim, psf, psf_dims, threshold) ;

    // Calculate the trimmed dims
    for (int64_t i = 0; i < 3; ++i)
        trimmed_psf_dims[i] = psf_dims[i] - 2 * n_elements_to_trim[i] ;

    // Calculate the number of elements to allocate
    int64_t trimmed_psf_element_count = element_count_from_dims_3d(trimmed_psf_dims) ;

    // Allocate it
    float* trimmed_psf = new float[trimmed_psf_element_count] ;

    // Now copy the data over
    copy_slice(trimmed_psf, trimmed_psf_dims, n_elements_to_trim, 3, psf, psf_dims) ;

    // Eliminate any negative elements that might have resulted from cubic interpolation
    pos_in_place_3d(trimmed_psf, trimmed_psf_dims) ;

    // Normalize the trimmedPSF
    normalize_in_place_3d(trimmed_psf, trimmed_psf_dims) ;

    // Return the result, hopefully with copy elision
    return trimmed_psf ;
}


//=============================================================
template float* fa_padArrayWithConstant<float>(const float* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims,float constant);
template std::uint16_t* fa_padArrayWithConstant<std::uint16_t>(const std::uint16_t* im,const std::int64_t *dimsNow,const std::uint32_t *dimsAfterPad,int ndims,uint16_t constant);
template std::uint8_t* fa_padArrayWithConstant<std::uint8_t>(const std::uint8_t* im,const std::int64_t *dimsNow,const std::uint32_t *dimsAfterPad,int ndims,uint8_t constant);
