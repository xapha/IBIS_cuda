#include "ibis.cuh"

//===========================================================================
///	RGB2LAB
//===========================================================================
__global__ void RGB2LAB( float* RGB, __c_ibis* __c_buffer, int count_exec ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( index >= count_exec )
        return;
    
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
    float r = float( RGB[ index * 3 + 2 ] ) / 255;
	float g = float( RGB[ index * 3 + 1 ] ) / 255;
    float b = float( RGB[ index * 3 ] ) / 255;
	
	float X = r*0.412453 + g*0.357580 + b*0.180423;
	float Y = r*0.212671 + g*0.715160 + b*0.072169;
	float Z = r*0.019334 + g*0.119193 + b*0.950227;
	
	//------------------------
	// XYZ to LAB conversion
	//------------------------
	float Xn = 0.950456;	//reference white
	float Zn = 1.088754;	//reference white
	
	X /= Xn;
	Z /= Zn;
	
	float epsilon = 0.008856;	//actual CIE standard
	float kappa   = 903.3;		//actual CIE standard

    
    float l, a;

    if( Y > epsilon )
        l = 116 * powf(Y, 1.0/3.0) - 16;
    else
        l = kappa * Y;

    float fx, fy, fz;
    if(X > epsilon)
        fx = powf(X, 1.0/3.0);
    else
        fx = 7.787*X + 16.0/116.0;

    if(Y > epsilon)
        fy = powf(Y, 1.0/3.0);
    else
        fy = 7.787*Y + 16.0/116.0;

    if(Z > epsilon)
        fz = powf(Z, 1.0/3.0);
    else
        fz = 7.787*Z + 16.0/116.0;
    
    a = 500 * ( fx - fy ) + 128;
    b = 500 * ( fy - fz ) + 128;
	
	__c_buffer->__lab[ 3*index + 0 ] = l * 255 / 100;
	__c_buffer->__lab[ 3*index + 1 ] = a + 128;
	__c_buffer->__lab[ 3*index + 2 ] = b + 128;
	
	//if( index == 26894 )
	//    printf(" -- gpu -- (%i, %i, %i) = (%f, %f, %f) \n ", int( R[ index ] ), int( G[ index ] ), int( B[ index ] ), __c_buffer->__l_vec[ index ], __c_buffer->__a_vec[ index ], __c_buffer->__b_vec[ index ] );
	
}
