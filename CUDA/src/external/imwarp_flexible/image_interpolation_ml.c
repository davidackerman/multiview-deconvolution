#include <math.h>
#include <stdbool.h>
#include "image_interpolation_ml.h"

#define EPS  1e-20
#define EPSF 1e-12f

/* power of integers */
static __inline double pow2(double val) { return val*val; }
static __inline double pow3(double val) { return val*val*val; }
//static __inline double pow4(double val) { return pow2(val)*pow2(val); }
//static __inline float pow2_float(float val) { return val*val; }
//static __inline float pow3_float(float val) { return val*val*val; }
//static __inline float pow4_float(float val) { return pow2_float(val)*pow2_float(val); }

/*
#ifdef __LCC__
static __inline float floorfloat(float val) { return (float)floor((double)val); }
#else
static __inline float floorfloat(float val) { return floorf(val); }
#endif
*/

/* Image and Volume interpolation 
 *
 * Function is written by D.Kroon University of Twente (June 2009)
 */
// These are adapted s.t. they assume images and stacks are organized Matlab-style:
// I.e. as you go along the serial array, the y index changes most rapidly, then the x index, then the z index

/* Get an pixel from an image, if outside image, black or nearest pixel */
float get_float_voxel_ml(int64_t i_x, int64_t i_y, int64_t i_z, int64_t n_x, int64_t n_y, int64_t n_z, float *I) {
    return I[i_z*n_x*n_y+i_x*n_y+i_y];  // Assumes Matlab-style stack organization (column-major)
}

float interpolate_3d_float_linear_black_ml(double x, double y, double z, int64_t *dims, float *stack) {
    double result;
    /*  Linear interpolation variables */
    int64_t xBas0, xBas1, yBas0, yBas1, zBas0, zBas1;
    double perc[8];
    double xCom, yCom, zCom;
    double xComi, yComi, zComi;
    double color[8]={0, 0, 0, 0, 0, 0, 0, 0};
    double x_floor, y_floor, z_floor;
    
	// Stack dimensions
	int64_t n_x = dims[1];
	int64_t n_y = dims[0];
	int64_t n_z = dims[2];

    x_floor=floor(x); y_floor=floor(y); z_floor=floor(z);
    
    /* Determine the coordinates of the pixel(s) which will be come the current pixel */
    /* (using linear interpolation) */
	xBas0 = (int64_t)x_floor; yBas0 = (int64_t)y_floor; zBas0 = (int64_t)z_floor;
    xBas1=xBas0+1;      yBas1=yBas0+1;      zBas1=zBas0+1;
    
    
    color[0]=0; color[1]=0; color[2]=0; color[3]=0;
    color[4]=0; color[5]=0; color[6]=0; color[7]=0;
    
    if((xBas0>=0)&&(xBas0<n_x)) {
        if((yBas0>=0)&&(yBas0<n_y)) {
            if((zBas0>=0)&&(zBas0<n_z)) {
                color[0]=get_float_voxel_ml(xBas0, yBas0, zBas0, n_x, n_y, n_z, stack);
            }
            if((zBas1>=0)&&(zBas1<n_z)) {
                color[1]=get_float_voxel_ml(xBas0, yBas0, zBas1, n_x, n_y, n_z, stack);
            }
        }
        if((yBas1>=0)&&(yBas1<n_y)) {
            if((zBas0>=0)&&(zBas0<n_z)) {
                color[2]=get_float_voxel_ml(xBas0, yBas1, zBas0, n_x, n_y, n_z, stack);
            }
            if((zBas1>=0)&&(zBas1<n_z)) {
                color[3]=get_float_voxel_ml(xBas0, yBas1, zBas1, n_x, n_y, n_z, stack);
            }
        }
    }
    if((xBas1>=0)&&(xBas1<n_x))  {
        if((yBas0>=0)&&(yBas0<n_y)) {
            if((zBas0>=0)&&(zBas0<n_z)) {
                color[4]=get_float_voxel_ml(xBas1, yBas0, zBas0, n_x, n_y, n_z, stack);
            }
            if((zBas1>=0)&&(zBas1<n_z)) {
                color[5]=get_float_voxel_ml(xBas1, yBas0, zBas1, n_x, n_y, n_z, stack);
            }
        }
        if((yBas1>=0)&&(yBas1<n_y)) {
            if((zBas0>=0)&&(zBas0<n_z)) {
                color[6]=get_float_voxel_ml(xBas1, yBas1, zBas0, n_x, n_y, n_z, stack);
            }
            if((zBas1>=0)&&(zBas1<n_z)) {
                color[7]=get_float_voxel_ml(xBas1, yBas1, zBas1, n_x, n_y, n_z, stack);
            }
        }
    }
    
    /* Linear interpolation constants (percentages) */
    xCom=x-x_floor;  yCom=y-y_floor;   zCom=z-z_floor;
    
    xComi=(1-xCom); yComi=(1-yCom); zComi=(1-zCom);
    perc[0]=xComi * yComi; perc[1]=perc[0] * zCom; perc[0]=perc[0] * zComi;
    perc[2]=xComi * yCom;  perc[3]=perc[2] * zCom; perc[2]=perc[2] * zComi;
    perc[4]=xCom * yComi;  perc[5]=perc[4] * zCom; perc[4]=perc[4] * zComi;
    perc[6]=xCom * yCom;   perc[7]=perc[6] * zCom; perc[6]=perc[6] * zComi;
    
    /* Set the current pixel value */
    result =color[0]*perc[0]+color[1]*perc[1]+color[2]*perc[2]+color[3]*perc[3]+color[4]*perc[4]+color[5]*perc[5]+color[6]*perc[6]+color[7]*perc[7];
    return (float) result ;
}

float interpolate_3d_float_linear_ml(double x, double y, double z, int64_t *dims, float *stack) {
    double Iout;
    /*  Linear interpolation variables */
	int64_t xBas0, xBas1, yBas0, yBas1, zBas0, zBas1;
    double perc[8];
    double xCom, yCom, zCom;
    double xComi, yComi, zComi;
    double color[8]={0, 0, 0, 0, 0, 0, 0, 0};
    double fTlocalx, fTlocaly, fTlocalz;
    
	// Stack dimensions
	int64_t n_x = dims[1];
	int64_t n_y = dims[0];
	int64_t n_z = dims[2];

	fTlocalx = floor(x); fTlocaly = floor(y); fTlocalz = floor(z);
    
    /* Determine the coordinates of the pixel(s) which will be come the current pixel */
    /* (using linear interpolation) */
	xBas0 = (int64_t)fTlocalx; yBas0 = (int64_t)fTlocaly; zBas0 = (int64_t)fTlocalz;
    xBas1=xBas0+1;      yBas1=yBas0+1;      zBas1=zBas0+1;
    
    /* Clamp to boundary */
    if(xBas0<0) {xBas0=0; if(xBas1<0) { xBas1=0; }}
    if(yBas0<0) {yBas0=0; if(yBas1<0) { yBas1=0; }}
    if(zBas0<0) {zBas0=0; if(zBas1<0) { zBas1=0; }}
    if(xBas1>(n_x-1)) { xBas1=n_x-1; if(xBas0>(n_x-1)) { xBas0=n_x-1; }}
    if(yBas1>(n_y-1)) { yBas1=n_y-1; if(yBas0>(n_y-1)) { yBas0=n_y-1; }}
    if(zBas1>(n_z-1)) { zBas1=n_z-1; if(zBas0>(n_z-1)) { zBas0=n_z-1; }}
    
    /*  Get intensities */
    color[0]=get_float_voxel_ml(xBas0, yBas0, zBas0, n_x, n_y, n_z, stack);
    color[1]=get_float_voxel_ml(xBas0, yBas0, zBas1, n_x, n_y, n_z, stack);
    color[2]=get_float_voxel_ml(xBas0, yBas1, zBas0, n_x, n_y, n_z, stack);
    color[3]=get_float_voxel_ml(xBas0, yBas1, zBas1, n_x, n_y, n_z, stack);
    color[4]=get_float_voxel_ml(xBas1, yBas0, zBas0, n_x, n_y, n_z, stack);
    color[5]=get_float_voxel_ml(xBas1, yBas0, zBas1, n_x, n_y, n_z, stack);
    color[6]=get_float_voxel_ml(xBas1, yBas1, zBas0, n_x, n_y, n_z, stack);
    color[7]=get_float_voxel_ml(xBas1, yBas1, zBas1, n_x, n_y, n_z, stack);
    
    /* Linear interpolation constants (percentages) */
    xCom=x-fTlocalx;  yCom=y-fTlocaly;   zCom=z-fTlocalz;
    
    xComi=(1-xCom); yComi=(1-yCom); zComi=(1-zCom);
    perc[0]=xComi * yComi; perc[1]=perc[0] * zCom; perc[0]=perc[0] * zComi;
    perc[2]=xComi * yCom;  perc[3]=perc[2] * zCom; perc[2]=perc[2] * zComi;
    perc[4]=xCom * yComi;  perc[5]=perc[4] * zCom; perc[4]=perc[4] * zComi;
    perc[6]=xCom * yCom;   perc[7]=perc[6] * zCom; perc[6]=perc[6] * zComi;
    
    /* Set the current pixel value */
    Iout =color[0]*perc[0]+color[1]*perc[1]+color[2]*perc[2]+color[3]*perc[3]+color[4]*perc[4]+color[5]*perc[5]+color[6]*perc[6]+color[7]*perc[7];
    return (float) Iout ;
}

float interpolate_3d_float_cubic_black_ml(double x, double y, double z, int64_t *dims, float *stack) {
    /* Floor of coordinate */
    double fTlocalx, fTlocaly, fTlocalz;
    /* Zero neighbor */
    int64_t xBas0, yBas0, zBas0;
    /* The location in between the pixels 0..1 */
    double tx, ty, tz;
    /* Neighbor loccations */
    int64_t xn[4], yn[4], zn[4];
    
    /* The vectors */
    double vector_tx[4], vector_ty[4], vector_tz[4];
    double vector_qx[4], vector_qy[4], vector_qz[4];
    /* Interpolated Intensity; */
    double Ipixelx=0, Ipixelxy=0, Ipixelxyz=0;
    /* Loop variable */
    int i, j;
    /* constant 0.5; */
    const double con=0.5;
    
	// Stack dimensions
	int64_t n_x = dims[1] ;
	int64_t n_y = dims[0];
	int64_t n_z = dims[2];

    /* Determine of the zero neighbor */
    fTlocalx=floor(x); fTlocaly=floor(y); fTlocalz=floor(z);
    xBas0=(int64_t) fTlocalx; yBas0=(int64_t) fTlocaly; zBas0=(int64_t) fTlocalz;
    
    /* Determine the location in between the pixels 0..1 */
    tx=x-fTlocalx; ty=y-fTlocaly; tz=z-fTlocalz;
    
    /* Determine the t vectors */
    vector_tx[0]= con; vector_tx[1]= con*tx; vector_tx[2]= con*pow2(tx); vector_tx[3]= con*pow3(tx);
    vector_ty[0]= con; vector_ty[1]= con*ty; vector_ty[2]= con*pow2(ty); vector_ty[3]= con*pow3(ty);
    vector_tz[0]= con; vector_tz[1]= con*tz; vector_tz[2]= con*pow2(tz); vector_tz[3]= con*pow3(tz);
    
    /* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
    vector_qx[0]= (double)-1.0*vector_tx[1]+(double)2.0*vector_tx[2]-(double)1.0*vector_tx[3];
    vector_qx[1]= (double)2.0*vector_tx[0]-(double)5.0*vector_tx[2]+(double)3.0*vector_tx[3];
    vector_qx[2]= (double)1.0*vector_tx[1]+(double)4.0*vector_tx[2]-(double)3.0*vector_tx[3];
    vector_qx[3]= (double)-1.0*vector_tx[2]+(double)1.0*vector_tx[3];
    vector_qy[0]= -(double)1.0*vector_ty[1]+(double)2.0*vector_ty[2]-(double)1.0*vector_ty[3];
    vector_qy[1]= (double)2.0*vector_ty[0]-(double)5.0*vector_ty[2]+(double)3.0*vector_ty[3];
    vector_qy[2]= (double)1.0*vector_ty[1]+(double)4.0*vector_ty[2]-(double)3.0*vector_ty[3];
    vector_qy[3]= -(double)1.0*vector_ty[2]+(double)1.0*vector_ty[3];
    vector_qz[0]= -(double)1.0*vector_tz[1]+(double)2.0*vector_tz[2]-(double)1.0*vector_tz[3];
    vector_qz[1]= (double)2.0*vector_tz[0]-(double)5.0*vector_tz[2]+(double)3.0*vector_tz[3];
    vector_qz[2]= (double)1.0*vector_tz[1]+(double)4.0*vector_tz[2]-(double)3.0*vector_tz[3];
    vector_qz[3]= -(double)1.0*vector_tz[2]+(double)1.0*vector_tz[3];
    
    /* Determine 1D neighbour coordinates */
    xn[0]=xBas0-1; xn[1]=xBas0; xn[2]=xBas0+1; xn[3]=xBas0+2;
    yn[0]=yBas0-1; yn[1]=yBas0; yn[2]=yBas0+1; yn[3]=yBas0+2;
    zn[0]=zBas0-1; zn[1]=zBas0; zn[2]=zBas0+1; zn[3]=zBas0+2;
    
    /* First do interpolation in the x direction followed by interpolation in the y direction */
    for(j=0; j<4; j++) {
        Ipixelxy=0;
        if((zn[j]>=0)&&(zn[j]<n_z)) {
            for(i=0; i<4; i++) {
                Ipixelx=0;
                if((yn[i]>=0)&&(yn[i]<n_y)) {
                    if((xn[0]>=0)&&(xn[0]<n_x)) {
                        Ipixelx+=vector_qx[0]*get_float_voxel_ml(xn[0], yn[i], zn[j], n_x, n_y, n_z, stack);
                    }
                    if((xn[1]>=0)&&(xn[1]<n_x)) {
                        Ipixelx+=vector_qx[1]*get_float_voxel_ml(xn[1], yn[i], zn[j], n_x, n_y, n_z, stack);
                    }
                    if((xn[2]>=0)&&(xn[2]<n_x)) {
                        Ipixelx+=vector_qx[2]*get_float_voxel_ml(xn[2], yn[i], zn[j], n_x, n_y, n_z, stack);
                    }
                    if((xn[3]>=0)&&(xn[3]<n_x)) {
                        Ipixelx+=vector_qx[3]*get_float_voxel_ml(xn[3], yn[i], zn[j], n_x, n_y, n_z, stack);
                    }
                }
                Ipixelxy += vector_qy[i]*Ipixelx;
            }
            Ipixelxyz += vector_qz[j]*Ipixelxy;
        }
    }
    return (float)Ipixelxyz ;
}
float interpolate_3d_float_cubic_ml(double x, double y, double z, int64_t *dims, float *stack) {
    /* Floor of coordinate */
    double fTlocalx, fTlocaly, fTlocalz;
    /* Zero neighbor */
    int64_t xBas0, yBas0, zBas0;
    /* The location in between the pixels 0..1 */
    double tx, ty, tz;
    /* Neighbor loccations */
    int64_t xn[4], yn[4], zn[4];
    
    /* The vectors */
    double vector_tx[4], vector_ty[4], vector_tz[4];
    double vector_qx[4], vector_qy[4], vector_qz[4];
    
    /* Interpolated Intensity; */
    double Ipixelx=0, Ipixelxy=0, Ipixelxyz=0;
    /* Temporary value boundary */
    int64_t b;
    /* Loop variable */
    int i, j;
    /* const 0.5; */
    const double con=0.5;
    
	// Stack dimensions
	int64_t n_x = dims[1];
	int64_t n_y = dims[0];
	int64_t n_z = dims[2];

	/* Determine of the zero neighbor */
    fTlocalx = floor(x); fTlocaly = floor(y); fTlocalz = floor(z);
	xBas0 = (int64_t)fTlocalx; yBas0 = (int64_t)fTlocaly; zBas0 = (int64_t)fTlocalz;
    
    /* Determine the location in between the pixels 0..1 */
    tx=x-fTlocalx; ty=y-fTlocaly; tz=z-fTlocalz;
    
    /* Determine the t vectors */
    vector_tx[0]= con; vector_tx[1]= con*tx; vector_tx[2]= con*pow2(tx); vector_tx[3]= con*pow3(tx);
    vector_ty[0]= con; vector_ty[1]= con*ty; vector_ty[2]= con*pow2(ty); vector_ty[3]= con*pow3(ty);
    vector_tz[0]= con; vector_tz[1]= con*tz; vector_tz[2]= con*pow2(tz); vector_tz[3]= con*pow3(tz);
    
    /* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
    /* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
    vector_qx[0]= (double)-1.0*vector_tx[1]+(double)2.0*vector_tx[2]-(double)1.0*vector_tx[3];
    vector_qx[1]= (double)2.0*vector_tx[0]-(double)5.0*vector_tx[2]+(double)3.0*vector_tx[3];
    vector_qx[2]= (double)1.0*vector_tx[1]+(double)4.0*vector_tx[2]-(double)3.0*vector_tx[3];
    vector_qx[3]= (double)-1.0*vector_tx[2]+(double)1.0*vector_tx[3];
    vector_qy[0]= -(double)1.0*vector_ty[1]+(double)2.0*vector_ty[2]-(double)1.0*vector_ty[3];
    vector_qy[1]= (double)2.0*vector_ty[0]-(double)5.0*vector_ty[2]+(double)3.0*vector_ty[3];
    vector_qy[2]= (double)1.0*vector_ty[1]+(double)4.0*vector_ty[2]-(double)3.0*vector_ty[3];
    vector_qy[3]= -(double)1.0*vector_ty[2]+(double)1.0*vector_ty[3];
    vector_qz[0]= -(double)1.0*vector_tz[1]+(double)2.0*vector_tz[2]-(double)1.0*vector_tz[3];
    vector_qz[1]= (double)2.0*vector_tz[0]-(double)5.0*vector_tz[2]+(double)3.0*vector_tz[3];
    vector_qz[2]= (double)1.0*vector_tz[1]+(double)4.0*vector_tz[2]-(double)3.0*vector_tz[3];
    vector_qz[3]= -(double)1.0*vector_tz[2]+(double)1.0*vector_tz[3];
    
    /* Determine 1D neighbour coordinates */
    xn[0]=xBas0-1; xn[1]=xBas0; xn[2]=xBas0+1; xn[3]=xBas0+2;
    yn[0]=yBas0-1; yn[1]=yBas0; yn[2]=yBas0+1; yn[3]=yBas0+2;
    zn[0]=zBas0-1; zn[1]=zBas0; zn[2]=zBas0+1; zn[3]=zBas0+2;
    
    /* Clamp to boundary */
    
    /* Clamp to boundary */
    if(xn[0]<0) { xn[0]=0;if(xn[1]<0) { xn[1]=0;if(xn[2]<0) { xn[2]=0; if(xn[3]<0) { xn[3]=0; }}}}
    if(yn[0]<0) { yn[0]=0;if(yn[1]<0) { yn[1]=0;if(yn[2]<0) { yn[2]=0; if(yn[3]<0) { yn[3]=0; }}}}
    if(zn[0]<0) { zn[0]=0;if(zn[1]<0) { zn[1]=0;if(zn[2]<0) { zn[2]=0; if(zn[3]<0) { zn[3]=0; }}}}
    b=n_x-1;
    if(xn[3]>b) { xn[3]=b;if(xn[2]>b) { xn[2]=b;if(xn[1]>b) { xn[1]=b; if(xn[0]>b) { xn[0]=b; }}}}
    b=n_y-1;
    if(yn[3]>b) { yn[3]=b;if(yn[2]>b) { yn[2]=b;if(yn[1]>b) { yn[1]=b; if(yn[0]>b) { yn[0]=b; }}}}
    b=n_z-1;
    if(zn[3]>b) { zn[3]=b;if(zn[2]>b) { zn[2]=b;if(zn[1]>b) { zn[1]=b; if(zn[0]>b) { zn[0]=b; }}}}
    
    /* First do interpolation in the x direction followed by interpolation in the y direction */
    for(j=0; j<4; j++) {
        Ipixelxy=0;
        for(i=0; i<4; i++) {
            Ipixelx=0;
            Ipixelx+=vector_qx[0]*get_float_voxel_ml(xn[0], yn[i], zn[j], n_x, n_y, n_z, stack);
            Ipixelx+=vector_qx[1]*get_float_voxel_ml(xn[1], yn[i], zn[j], n_x, n_y, n_z, stack);
            Ipixelx+=vector_qx[2]*get_float_voxel_ml(xn[2], yn[i], zn[j], n_x, n_y, n_z, stack);
            Ipixelx+=vector_qx[3]*get_float_voxel_ml(xn[3], yn[i], zn[j], n_x, n_y, n_z, stack);
            Ipixelxy+= vector_qy[i]*Ipixelx;
        }
        Ipixelxyz+= vector_qz[j]*Ipixelxy;
    }
    return (float)Ipixelxyz ;
}

float interpolate_3d_float_gray_ml(double x, double y, double z, int64_t *dims, float *stack, bool is_cubic, bool is_background_black)  {
	float result;
	if (is_cubic) {
		if (is_background_black) { result = interpolate_3d_float_cubic_black_ml(x, y, z, dims, stack); }
        else { result = interpolate_3d_float_cubic_ml(x, y, z, dims, stack); }
	}
	else {
        if (is_background_black) { result = interpolate_3d_float_linear_black_ml(x, y, z, dims, stack); }
        else { result = interpolate_3d_float_linear_ml(x, y, z, dims, stack); }
	}
	return result;
}

