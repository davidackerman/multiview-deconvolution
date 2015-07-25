#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#include "math.h"
#include <stdbool.h>
#include <stdio.h>
#include "affine_transform_3d_single.h"
#include "image_interpolation.h"
#include "multiple_os_thread.h"


/* This function transforms a volume with a 4x4 transformation matrix
 *
 * affine transform Iout=affine_transform_3d_double(Iin,Minv,mode)
 *
 * inputs,
 *  Iin: The greyscale 3D input image
 *  Minv: The (inverse) 4x4 transformation matrix
 *  mode: If 0: linear interpolation and outside pixels set to nearest pixel
 *           1: linear interpolation and outside pixels set to zero
 *           2: cubic interpolation and outsite pixels set to nearest pixel
 *           3: cubic interpolation and outside pixels set to zero
 * output,
 *  Iout: The transformed 3D image
 *
 *
 * 
 * Function is written by D.Kroon University of Twente (June 2009)
 */



//from http://www.cprogramming.com/snippets/source-code/find-the-number-of-cpu-cores-for-windows-mac-or-linux
int getNumberOfCores() 
{
#ifdef WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	return sysinfo.dwNumberOfProcessors;
#elif MACOS
	int nm[2];
	size_t len = 4;
	uint32_t count;

	nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
	sysctl(nm, 2, &count, &len, NULL, 0);

	if (count < 1) {
		nm[1] = HW_NCPU;
		sysctl(nm, 2, &count, &len, NULL, 0);
		if (count < 1) { count = 1; }
	}
	return count;
#else
	return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

//================================================================================

voidthread transformvolume(float **Args) {
    float *Isize_d, *mean, *A, *Iin, *Iout, *ThreadID, *moded;
    int Isize[3]={0, 0, 0};
    int mode=0;
    int x, y, z;
    float *Nthreadsd;
    int Nthreads;
    bool black, cubic;   
    
    // Location of pixel which will be come the current pixel */
    float Tlocalx, Tlocaly, Tlocalz;
    
    // X,Y,Z coordinates of current pixel */
    float xd, yd, zd;
    
    // Variables to store 1D index */
    int indexI;
    
    // Multiple threads, one does the odd the other even indexes */
    int ThreadOffset;
    
    // Split up matrix multiply to make registration process faster  */
    float acomp0, acomp1, acomp2;
    float bcomp0, bcomp1, bcomp2;
    float ccomp0, ccomp1, ccomp2;
    
    Isize_d=Args[0];
    mean=Args[1];
    A=Args[2];
    Iin=Args[3];
    Iout=Args[4];
    ThreadID=Args[5];
    moded=Args[6]; 
	mode=(int)moded[0];
    Nthreadsd=Args[7];  
	Nthreads=(int)Nthreadsd[0];
                
    if(mode==0||mode==2){ black = false; } else { black = true; }
    if(mode==0||mode==1){ cubic = false; } else { cubic = true; }
	
    Isize[0] = (int)Isize_d[0];
    Isize[1] = (int)Isize_d[1];
    Isize[2] = (int)Isize_d[2];
    
    ThreadOffset=(int) ThreadID[0];
    
    acomp0=mean[0] + A[3]; acomp1=mean[1] + A[7]; acomp2=mean[2] + A[11];
    //  Loop through all image pixel coordinates */
    for (z=ThreadOffset; z<Isize[2]; z=z+Nthreads) {
        zd=z-mean[2];
        bcomp0 = A[2] *zd + acomp0;
        bcomp1 = A[6] *zd + acomp1;
        bcomp2 = A[10]*zd + acomp2;
        for (y=0; y<Isize[1]; y++) {
            yd=y-mean[1];
            ccomp0 = A[1] *yd + bcomp0;
            ccomp1 = A[5] *yd + bcomp1;
            ccomp2 = A[9] *yd + bcomp2;
            for (x=0; x<Isize[0]; x++) {
                xd=x-mean[0];
                Tlocalx = A[0] * xd + ccomp0;
                Tlocaly = A[4] * xd + ccomp1;
                Tlocalz = A[8] * xd + ccomp2;
                
                indexI=mindex3(x, y, z, Isize[0], Isize[1]);
                  
                // the pixel interpolation */
                Iout[indexI]=interpolate_3d_float_gray(Tlocalx, Tlocaly, Tlocalz, Isize, Iin, cubic, black);
            }
        }
    }
    
    //  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
    EndThread;
}

//============================================================
void affine_3d_transpose(const float  A[AFFINE_3D_MATRIX_SIZE], float Atr[AFFINE_3D_MATRIX_SIZE])
{
	int ii, jj;
	for (ii = 0; ii < 4; ii++)
	{
		for (jj = 0; jj < 4; jj++)
		{
			Atr[4 * ii + jj] = A[4 * jj + ii];
		}
	}
}
void affine_3d_compose(const float  A[AFFINE_3D_MATRIX_SIZE], const float B[AFFINE_3D_MATRIX_SIZE], float C[AFFINE_3D_MATRIX_SIZE])
{
	int ii, jj, kk;
	float aux;
	for (ii = 0; ii < 4; ii++)//columns
	{
		for (jj = 0; jj < 4; jj++)//rows
		{
			aux = 0;
			for (kk = 0; kk < 4; kk++)
				aux += A[4 * kk + jj] * B[4 * ii + kk];

			C[4 * ii + jj] = aux;
		}
	}
}

#define _m(x,y) (A[x + 4 * y])
#define _minv(x,y) (Ainv[x + 4 * y])
void affine_3d_inverse(const float  A[AFFINE_3D_MATRIX_SIZE], float Ainv[AFFINE_3D_MATRIX_SIZE])
{
	if (affine_3d_isAffine(A) == false)
	{
		printf("ERROR: affine_3d_inverse: trying to calculate inverse of a non-affine matrix\n");
		exit(2);
	}

	//A = [ [R , zeros(3,1)]; [Tx Ty Tz 1] ]
	//Ainv = [ [Rinv , zeros(3,1)]; [-[Tx Ty Tz]*Rinv 1] ]

	//set easy row
	Ainv[12] = Ainv[13] = Ainv[14] = 0.0f;
	Ainv[15] = 1.0f;


	// computes the inverse of a matrix m
	double det = _m(0, 0) * (_m(1, 1) * _m(2, 2) - _m(2, 1) * _m(1, 2)) -
		_m(0, 1) * (_m(1, 0) * _m(2, 2) - _m(1, 2) * _m(2, 0)) +
		_m(0, 2) * (_m(1, 0) * _m(2, 1) - _m(1, 1) * _m(2, 0));

	if ( det < 1e-8 )
	{
		printf("ERROR: affine_3d_inverse: matrix is close to singular\n");
		exit(2);
	}

	double invdet = 1 / det;

	_minv(0, 0) = (_m(1, 1) * _m(2, 2) - _m(2, 1) * _m(1, 2)) * invdet;
	_minv(0, 1) = (_m(0, 2) * _m(2, 1) - _m(0, 1) * _m(2, 2)) * invdet;
	_minv(0, 2) = (_m(0, 1) * _m(1, 2) - _m(0, 2) * _m(1, 1)) * invdet;
	_minv(1, 0) = (_m(1, 2) * _m(2, 0) - _m(1, 0) * _m(2, 2)) * invdet;
	_minv(1, 1) = (_m(0, 0) * _m(2, 2) - _m(0, 2) * _m(2, 0)) * invdet;
	_minv(1, 2) = (_m(1, 0) * _m(0, 2) - _m(0, 0) * _m(1, 2)) * invdet;
	_minv(2, 0) = (_m(1, 0) * _m(2, 1) - _m(2, 0) * _m(1, 1)) * invdet;
	_minv(2, 1) = (_m(2, 0) * _m(0, 1) - _m(0, 0) * _m(2, 1)) * invdet;
	_minv(2, 2) = (_m(0, 0) * _m(1, 1) - _m(1, 0) * _m(0, 1)) * invdet;

	//translation
	Ainv[3] = -(A[3] * _minv(0, 0) + A[7] * _minv(1, 0) + A[11] * _minv(2, 0));
	Ainv[7] = -(A[3] * _minv(0, 1) + A[7] * _minv(1, 1) + A[11] * _minv(2, 1));
	Ainv[11] = -(A[3] * _minv(0, 2) + A[7] * _minv(1, 2) + A[11] * _minv(2, 2));
}

bool affine_3d_isAffine(const float A[AFFINE_3D_MATRIX_SIZE])
{	
	//check the last columns. It should be [0, 0, 0, 1]'
	if (A[12] != 0.0f)
		return false;
	if (A[13] != 0.0f)
		return false;
	if (A[14] != 0.0f)
		return false;
	if (A[15] != 1.0f)
		return false;

	return true;
}


void affine3d_printMatrix(const float A[AFFINE_3D_MATRIX_SIZE])
{
	for (int ii = 0; ii < 4; ii++)
		printf("%.6f\t%.6f\t%.6f\t%.6f\n", A[ii + 4 * 0], A[ii + 4 * 1], A[ii + 4 * 2], A[ii + 4 * 3] );
}

//============================================================
void imwarpFast_MatlabEquivalent(const float* imIn, float* imOut, int64_t dimsIn[3], int64_t dimsOut[3], float A[AFFINE_3D_MATRIX_SIZE], int interpMode)
{
	printf("===============TODO=================\n");
	printf("DONE!! 1.-Write test for affine operations (do not use diagonal matrices)\n");
	printf("2.-Write this function looking at the Matlab equivalent\n");
	printf("3.-Test it with a real image using a real transformation for camera 1,2 or 3");
}

//================================================================================================================
//Function that replaces the original mex function
void affineTransform_3d_float(const float* Iin, float* Iout, int64_t dims[3], float A[AFFINE_3D_MATRIX_SIZE], int interpMode)
{

	printf("=======DEBUGGING:affineTransform_3d_float: I might need to transpose A at the very beginning===============");

	float *moded, *Nthreadsf;		
	int Nthreads;

	// float pointer array to store all needed function variables  
	float ***ThreadArgs;
	float **ThreadArgs1;

	// Handles to the worker threads 
	ThreadHANDLE *ThreadList;


	// ID of Threads 
	float **ThreadID;
	float *ThreadID1;

	// Loop variable  
	int i;


	// Size of input image 
	float Isize_d[3] = { 0, 0, 0 };

	float mean[3] = { 0, 0, 0 };


	// Get the sizes of the image 		
	Isize_d[0] = (float)dims[0]; 
	Isize_d[1] = (float)dims[1]; 
	Isize_d[2] = (float)dims[2];
	

	// Assign pointers to each input. 	
	*moded = ((float)(interpMode));

	

	
	Nthreads = getNumberOfCores();
	*Nthreadsf = ((float)(Nthreads));

	// Reserve room for handles of threads in ThreadList  
	ThreadList = (ThreadHANDLE*)malloc(Nthreads* sizeof(ThreadHANDLE));


	ThreadID = (float **)malloc(Nthreads* sizeof(float *));
	ThreadArgs = (float ***)malloc(Nthreads* sizeof(float **));


	// Center of the volume 
	mean[0] = Isize_d[0] / 2;  
	mean[1] = Isize_d[1] / 2;  
	mean[2] = Isize_d[2] / 2;

	for (i = 0; i<Nthreads; i++) 
	{
		//  Make Thread ID  
		ThreadID1 = (float *)malloc(1 * sizeof(float));
		ThreadID1[0] = (float)i;
		ThreadID[i] = ThreadID1;

		//  Make Thread Structure  
		ThreadArgs1 = (float **)malloc(8 * sizeof(float *));
		ThreadArgs1[0] = Isize_d;
		ThreadArgs1[1] = mean;
		ThreadArgs1[2] = A;
		ThreadArgs1[3] = Iin;
		ThreadArgs1[4] = Iout;
		ThreadArgs1[5] = ThreadID[i];
		ThreadArgs1[6] = moded;
		ThreadArgs1[7] = Nthreadsf;
		// Start a Thread  
		ThreadArgs[i] = ThreadArgs1;
		StartThread(ThreadList[i], &transformvolume, ThreadArgs[i])
	}


	for (i = 0; i<Nthreads; i++) { WaitForThreadFinish(ThreadList[i]); }



	for (i = 0; i<Nthreads; i++) 
	{
		free(ThreadArgs[i]);
		free(ThreadID[i]);
	}

	free(ThreadArgs);
	free(ThreadID);
	free(ThreadList);
}

/*
// The matlab mex function
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] ) {
    // Ox and Oy are the grid points 
    // Zo is the input image 
    // Zi is the transformed image 
    // nx and ny are the number of grid points (inside the image) 
    float *Iin, *Iout, *M, *moded;
    mxArray *matlabCallOut[1]={0};
    mxArray *matlabCallIn[1]={0};
    double *Nthreadsd; float Nthreadsf[1]={0};
    int Nthreads;
    
    // float pointer array to store all needed function variables  
    float ***ThreadArgs;
    float **ThreadArgs1;
    
	// Handles to the worker threads 

		ThreadHANDLE *ThreadList;

    
    // ID of Threads 
    float **ThreadID;
    float *ThreadID1;
    
    // Loop variable  
    int i;
    
    // Transformation matrix 
    float A[16]={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    // Size of input image 
    float Isize_d[3]={0, 0, 0};
    const mwSize *dims;
    
    float mean[3]={0, 0, 0};
    
    // Check for proper number of arguments. 
    if(nrhs!=3) {
        mexErrMsgTxt("Three inputs are required.");
    } else if(nlhs!=1) {
        mexErrMsgTxt("One output required");
    }
    // nsubs=mxGetNumberOfDimensions(prhs[0]);  
    
    // Get the sizes of the image 
    dims = mxGetDimensions(prhs[0]);
    Isize_d[0] = (float)dims[0]; Isize_d[1] = (float)dims[1]; Isize_d[2] = (float)dims[2];
    
    // Create output array 
    plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    
    // Assign pointers to each input. 
    Iin=(float *)mxGetData(prhs[0]);
    M=(float *)mxGetData(prhs[1]);
    moded=(float *)mxGetData(prhs[2]);
    
    A[0] = M[mindex2(0, 0, 4)];  A[1] = M[mindex2(0, 1, 4)];  A[2] = M[mindex2(0, 2, 4)];  A[3] = M[mindex2(0, 3, 4)];
    A[4] = M[mindex2(1, 0, 4)];  A[5] = M[mindex2(1, 1, 4)];  A[6] = M[mindex2(1, 2, 4)];  A[7] = M[mindex2(1, 3, 4)];
    A[8] = M[mindex2(2, 0, 4)];  A[9] = M[mindex2(2, 1, 4)];  A[10] = M[mindex2(2, 2, 4)]; A[11] = M[mindex2(2, 3, 4)];
    A[12] = M[mindex2(3, 0, 4)]; A[13] = M[mindex2(3, 1, 4)]; A[14] = M[mindex2(3, 2, 4)]; A[15] = M[mindex2(3, 3, 4)];
    
    mexCallMATLAB(1, matlabCallOut, 0, matlabCallIn, "maxNumCompThreads");
    Nthreadsd=mxGetPr(matlabCallOut[0]);  Nthreadsf[0]= (float)Nthreadsd[0];
    Nthreads=(int)Nthreadsd[0];
    // Reserve room for handles of threads in ThreadList  

		ThreadList = (ThreadHANDLE*)malloc(Nthreads* sizeof( ThreadHANDLE ));

	
    ThreadID = (float **)malloc( Nthreads* sizeof(float *) );
    ThreadArgs = (float ***)malloc( Nthreads* sizeof(float **) );
    
    // Assign pointer to output. 
    Iout = (float *)mxGetData(plhs[0]);
    
    // Center of the volume 
    mean[0]=Isize_d[0]/2;  mean[1]=Isize_d[1]/2;  mean[2]=Isize_d[2]/2;
    
    for (i=0; i<Nthreads; i++) {
        //  Make Thread ID  
        ThreadID1= (float *)malloc( 1* sizeof(float) );
        ThreadID1[0]=(float)i;
        ThreadID[i]=ThreadID1;
        
        //  Make Thread Structure  
        ThreadArgs1 = (float **)malloc( 8* sizeof( float * ) );
        ThreadArgs1[0]=Isize_d;
        ThreadArgs1[1]=mean;
        ThreadArgs1[2]=A;
        ThreadArgs1[3]=Iin;
        ThreadArgs1[4]=Iout;
        ThreadArgs1[5]=ThreadID[i];
        ThreadArgs1[6]=moded;
        ThreadArgs1[7]=Nthreadsf;
        // Start a Thread  
        ThreadArgs[i]=ThreadArgs1;
		StartThread(ThreadList[i], &transformvolume, ThreadArgs[i])
    }
    

    for (i=0; i<Nthreads; i++) { WaitForThreadFinish(ThreadList[i]); }
  
    
    
    for (i=0; i<Nthreads; i++) {
        free(ThreadArgs[i]);
        free(ThreadID[i]);
    }
    
    free(ThreadArgs);
    free(ThreadID );
    free(ThreadList);
}

*/


