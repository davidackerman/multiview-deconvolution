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
#include <stdlib.h>
#include <stdint.h>
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
    float *Isize_out_d, *Isize_in_d, *center_out, *center_in, *A, *Iin, *Iout, *ThreadID, *moded;
    int64_t Isize_out[3]={0, 0, 0};
    int64_t Isize_in[3] = { 0, 0, 0 };
    int mode = 0;
    int64_t x_out, y_out, z_out;
    float *Nthreadsd;
    int Nthreads;
    bool black, cubic;   
    
    // Location of pixel which will be come the current pixel */
    float x_in, y_in, z_in;
    
    // X,Y,Z coordinates of current pixel */
    float xd_out, yd_out, zd_out;
    
    // Variables to store 1D index */
    int64_t indexI;
    
    // Multiple threads, one does the odd the other even indexes */
    int ThreadOffset;
    
    // Split up matrix multiply to make registration process faster  */
    float acomp0, acomp1, acomp2;
    float bcomp0, bcomp1, bcomp2;
    float ccomp0, ccomp1, ccomp2;
    
    Isize_out_d=Args[0];
    center_out=Args[1];
    A=Args[2];
    Iin=Args[3];
    Iout=Args[4];
    ThreadID=Args[5];
    moded=Args[6]; 
	mode=(int)moded[0];
    Nthreadsd=Args[7];  
	Nthreads=(int)Nthreadsd[0];
    Isize_in_d = Args[8];
    center_in = Args[9];

    if(mode==0||mode==2){ black = false; } else { black = true; }
    if(mode==0||mode==1){ cubic = false; } else { cubic = true; }
	
    Isize_out[0] = (int64_t)Isize_out_d[0];
    Isize_out[1] = (int64_t)Isize_out_d[1];
    Isize_out[2] = (int64_t)Isize_out_d[2];

    Isize_in[0] = (int64_t)Isize_in_d[0];
    Isize_in[1] = (int64_t)Isize_in_d[1];
    Isize_in[2] = (int64_t)Isize_in_d[2];

    ThreadOffset=(int) ThreadID[0];
    
    acomp0 = center_in[0] + A[3]; 
    acomp1 = center_in[1] + A[7]; 
    acomp2 = center_in[2] + A[11];
    //  Loop through all image pixel coordinates */
    for (z_out=ThreadOffset; z_out<Isize_out[2]; z_out=z_out+Nthreads) {
        zd_out=z_out-center_out[2];  // this center_out is the center_out of the output space
        bcomp0 = A[2] *zd_out + acomp0;
        bcomp1 = A[6] *zd_out + acomp1;
        bcomp2 = A[10]*zd_out + acomp2;
        for (y_out=0; y_out<Isize_out[1]; y_out++) {
            yd_out = y_out - center_out[1];  // this center_out is the center_out of the output space
            ccomp0 = A[1] *yd_out + bcomp0;
            ccomp1 = A[5] *yd_out + bcomp1;
            ccomp2 = A[9] *yd_out + bcomp2;
            for (x_out=0; x_out<Isize_out[0]; x_out++) {
                xd_out = x_out - center_out[0];  // this center_out is the center_out of the output space
                x_in = A[0] * xd_out + ccomp0;
                y_in = A[4] * xd_out + ccomp1;
                z_in = A[8] * xd_out + ccomp2;
                
                indexI=mindex3(x_out, y_out, z_out, Isize_out[0], Isize_out[1]);
                  
                // the pixel interpolation */
                Iout[indexI]=interpolate_3d_float_gray(x_in, y_in, z_in, Isize_in, Iin, cubic, black);
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
		affine3d_printMatrix(A);
		exit(2);
	}

	//A = [ [R , zeros(3,1)]; [Tx Ty Tz 1] ]
	//Ainv = [ [Rinv , zeros(3,1)]; [-[Tx Ty Tz]*Rinv 1] ]

	


	// computes the inverse of a matrix m
	double det = _m(0, 0) * (_m(1, 1) * _m(2, 2) - _m(2, 1) * _m(1, 2)) -
		_m(0, 1) * (_m(1, 0) * _m(2, 2) - _m(1, 2) * _m(2, 0)) +
		_m(0, 2) * (_m(1, 0) * _m(2, 1) - _m(1, 1) * _m(2, 0));

	if ( fabs(det) < 1e-8 )
	{
		printf("ERROR: affine_3d_inverse: matrix is close to singular\n");
		affine3d_printMatrix(A);
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

	
	//translation for bottom row
	Ainv[3] = -(A[3] * _minv(0, 0) + A[7] * _minv(1, 0) + A[11] * _minv(2, 0));
	Ainv[7] = -(A[3] * _minv(0, 1) + A[7] * _minv(1, 1) + A[11] * _minv(2, 1));
	Ainv[11] = -(A[3] * _minv(0, 2) + A[7] * _minv(1, 2) + A[11] * _minv(2, 2));
	
	//translation for last column
	Ainv[12] = -(A[12] * _minv(0, 0) + A[13] * _minv(0, 1) + A[14] * _minv(0, 2));
	Ainv[13] = -(A[12] * _minv(1, 0) + A[13] * _minv(1, 1) + A[14] * _minv(1, 2));
	Ainv[14] = -(A[12] * _minv(2, 0) + A[13] * _minv(2, 1) + A[14] * _minv(2, 2));

	//set last element	
	Ainv[15] = 1.0f;
}

bool affine_3d_isAffine(const float A[AFFINE_3D_MATRIX_SIZE])
{	
	//check the last columns. It should be [0, 0, 0, 1]'
	if (A[12] != 0.0f || A[13] != 0.0f || A[14] != 0.0f || A[15] != 1.0f)
	{
		if (A[3] != 0.0f || A[7] != 0.0f || A[11] != 0.0f || A[15] != 1.0f)
		{
			return false;
		}
	}

	return true;
}


void affine3d_printMatrix(const float A[AFFINE_3D_MATRIX_SIZE])
{
	int ii;
	for (ii = 0; ii < 4; ii++)
		printf("%.6f\t%.6f\t%.6f\t%.6f\n", A[ii + 4 * 0], A[ii + 4 * 1], A[ii + 4 * 2], A[ii + 4 * 3] );
}

//============================================================
void imwarpFast_MatlabEquivalent(const float* imIn, float* imOut, int64_t dimsIn[3], int64_t dimsOut[3], float A[AFFINE_3D_MATRIX_SIZE], int interpMode)
{
	//matrix to flip xy coodinates
	const float F[AFFINE_3D_MATRIX_SIZE] = { 0.0f, 1.0f, 0.0f, 0.0f, 
                                             1.0f, 0.0f, 0.0f, 0.0f, 
                                             0.0f, 0.0f, 1.0f, 0.0f, 
                                             0.0f, 0.0f, 0.0f, 1.0f };
	//recenter original to apply transformation
	float B[AFFINE_3D_MATRIX_SIZE] = { 1.0f, 0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f, 0.0f, 
                                       0.0f, 0.0f, 1.0f, 0.0f, 
                                       0.0f, 0.0f, 0.0f, 1.0f };
	float C[AFFINE_3D_MATRIX_SIZE] = { 1.0f, 0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f, 0.0f, 
                                       0.0f, 0.0f, 1.0f, 0.0f, 
                                       0.0f, 0.0f, 0.0f, 1.0f };
	float Af[AFFINE_3D_MATRIX_SIZE];
	float Aaux[AFFINE_3D_MATRIX_SIZE];
	float* imInPadded;
	int ii;
	int64_t imSize;
	

	//check if the transformation is the identity. Then we only need to copy the file
	bool isId = false;
    /*
	for (ii = 0; ii < AFFINE_3D_MATRIX_SIZE; ii++)
	{
		if (fabs(A[ii] - B[ii]) > 1e-3)
		{
			isId = false;
			break;
		}
	}
    */

    // Define a matrix that does a translation, then undoes that translation at the end.  So C==inv(B)
	for (ii = 0; ii < 3; ii++)
        {
        //B[ii + 12] = 0.5f * dimsOut[ii] + 1.0f  ;
        //C[ii + 12] = - B[ii + 12] ;
        B[ii + 12] =   0.5f * dimsIn[ii] + 1.0f  ;  // might have to swap these
        C[ii + 12] = -(0.5f * dimsOut[ii] + 1.0f) ;
        }


	//resize input image if necessary
	bool imResize = false;
	for (ii = 0; ii < 3; ii++)
        {
		if (dimsOut[ii] < dimsIn[ii])
            {
			//printf("ERROR: code is not ready for output image size smaller than input image size\n");
            printf("WARNING: Code might not work right when output image size is smaller than input image size\n");
            }
        else if (dimsOut[ii] > dimsIn[ii])
            {
            /*imResize = true;*/
            imResize = false ;
            printf("WARNING: Code might not work right when output image size is larger than input image size\n");
            }
        }

	if (imResize)
	{
        // Pad the input array with zeros, thus imInPadded will be of size dimsOut
		imInPadded = fa_padArrayWithZeros(imIn, dimsIn, dimsOut, 3);
	}
	else{
		imInPadded = imIn;
	}

	if (isId)
	{		
		imSize = 1;
		for (ii = 0; ii < 3; ii++)
			imSize *= dimsOut[ii];
		memcpy(imOut, imInPadded, sizeof(float)* imSize);		
		return;
	}
	else{

		//apply transformations to A	
		affine_3d_transpose(A, Af);  // After, Af == A'
		//memcpy(Af, A, sizeof(float)* AFFINE_3D_MATRIX_SIZE);
		affine_3d_compose(Af, F, Aaux);  // After, Aaux == A'*F
		affine_3d_inverse(Aaux, Af);  // After, Af == inv(A'*F)
		affine_3d_compose(Af, F, Aaux);  // After, Aaux == inv(A'*F) * F == inv(F) * inv(A') * F

		//recenter transformation
		affine_3d_compose(C, Aaux, Af);  // After, Af == C * inv(F) * inv(A') * F
		affine_3d_compose(Af, B, Aaux);  // After, Aaux ==  C * inv(F) * inv(A') * F * B


		//apply transformation	
		affine_3d_transpose(Aaux, Af);  // Apply transposition to get it in the right order for the C library
          // After, Af ==  (C * inv(F) * inv(A') * F * B)' == B' * F' * (inv(A'))' * (inv(F))' * C' == B' * F' * inv(A) * inv(F') * C'
          // x = B' * F' * inv(A) * inv(F') * C' * y
		affineTransform_3d_float(imInPadded, dimsIn, imOut, dimsOut, Af, interpMode);
	}

	//release memory
	if (imResize)
		free(imInPadded);
}

//================================================================================================================
//Function that replaces the original mex function
void affineTransform_3d_float(const float* Iin, int64_t dims_in[3], float* Iout, int64_t dims_out[3], float A[AFFINE_3D_MATRIX_SIZE], int interpMode)
{

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
	float Isize_out_d[3] = { 0, 0, 0 };
    float Isize_in_d[3] = { 0, 0, 0 };

	float center_out[3] = { 0, 0, 0 };
    float center_in[3] = { 0, 0, 0 };


	// Get the sizes of the image 		
	Isize_out_d[0] = (float)dims_out[0]; 
	Isize_out_d[1] = (float)dims_out[1]; 
	Isize_out_d[2] = (float)dims_out[2];
	

    // Get the sizes of the image 		
    Isize_in_d[0] = (float)dims_in[0];
    Isize_in_d[1] = (float)dims_in[1];
    Isize_in_d[2] = (float)dims_in[2];


    // Assign pointers to each input. 	
	moded = (float*)malloc(sizeof(float));
	*moded = ((float)(interpMode));

	

	Nthreadsf = (float*)malloc(sizeof(float));
	Nthreads = getNumberOfCores();
	*Nthreadsf = ((float)(Nthreads));

	// Reserve room for handles of threads in ThreadList  
	ThreadList = (ThreadHANDLE*)malloc(Nthreads* sizeof(ThreadHANDLE));


	ThreadID = (float **)malloc(Nthreads* sizeof(float *));
	ThreadArgs = (float ***)malloc(Nthreads* sizeof(float **));


	// Center of the volume 
	center_out[0] = Isize_out_d[0] / 2;  
	center_out[1] = Isize_out_d[1] / 2;  
	center_out[2] = Isize_out_d[2] / 2;

    // Center of the volume 
    center_in[0] = Isize_in_d[0] / 2;
    center_in[1] = Isize_in_d[1] / 2;
    center_in[2] = Isize_in_d[2] / 2;

	for (i = 0; i<Nthreads; i++) 
	{
		//  Make Thread ID  
		ThreadID1 = (float *)malloc(1 * sizeof(float));
		ThreadID1[0] = (float)i;
		ThreadID[i] = ThreadID1;

		//  Make Thread Structure  
		ThreadArgs1 = (float **)malloc(10 * sizeof(float *));
		ThreadArgs1[0] = Isize_out_d;
		ThreadArgs1[1] = center_out;
		ThreadArgs1[2] = A;
		ThreadArgs1[3] = Iin;
		ThreadArgs1[4] = Iout;
		ThreadArgs1[5] = ThreadID[i];
		ThreadArgs1[6] = moded;
		ThreadArgs1[7] = Nthreadsf;
        ThreadArgs1[8] = Isize_in_d;
        ThreadArgs1[9] = center_in;
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
	free(Nthreadsf);
	free(moded);
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


float* fa_padArrayWithZeros(const float* im, const int64_t *dimsNow, const int64_t *dimsAfterPad, int ndims)
{
	int ii;
	int64_t nImg = 1, zz, yy;
	if (ndims != 3)
	{
		printf("TODO:ERROR: padArrayWithZeros: function not ready for other values of ndims except for 3\n");
		exit(3);
	}
	
	for (ii = 0; ii < ndims; ii++)
	{
		if (dimsNow[ii] > dimsAfterPad[ii])
		{
			printf("ERROR: padArrayWithZeros: new dimensions are smaller than current dimensions\n");
			return NULL;
		}
		nImg *= (int64_t)(dimsAfterPad[ii]);
	}

	
	float* p = (float*) calloc(nImg, sizeof(float));//calloc already initializes to zero

	//copy "lines" of x	
	size_t lineSize = dimsNow[0] * sizeof(float);
	int64_t idx = 0;
	int64_t count = 0;
	for (zz = 0; zz < dimsNow[2]; zz++)
	{
		idx = dimsAfterPad[0] * dimsAfterPad[1] * zz;
		for (yy = 0; yy < dimsNow[1]; yy++)
		{
			//update for new array
			//idx = dimsAfterPad[0] * ( yy + dimsAfterPad[1] * zz);
			//update for new array
			//count = dimsNow[0] * (yy + dimsNow[1] * zz);

			//copy elements
			memcpy(&(p[idx]), &(im[count]), lineSize);

			//update counters
			idx += dimsAfterPad[0];
			count += dimsNow[0];
		}
	}

	return p;
}
