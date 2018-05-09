#include "ibis.cuh"

//===========================================================================
///	RGB2LAB
//===========================================================================
__global__ void RGB2LAB( float* R, float* G, float* B, __c_ibis* __c_buffer, int count_exec ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    //printf(" %i / %i ", index, count_exec);
    
    if( index >= count_exec )
        return;
    
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double sR = double( R[ index ] ) / 255;
	double sG = double( G[ index ] ) / 255;
	double sB = double( B[ index ] ) / 255;
	
	double X, Y, Z;
	double r, g, b;

	if(sR <= 0.04045)
	    r = sR/12.92;
	else
	    r = powf((sR+0.055)/1.055,2.4);
	
	if(sG <= 0.04045)
	    g = sG/12.92;
	else
	    g = powf((sG+0.055)/1.055,2.4);
	
	if(sB <= 0.04045)
	    b = sB/12.92;
	else
	    b = powf((sB+0.055)/1.055,2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
	
	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon)
	    fx = powf(xr, 1.0/3.0);
	else
	    fx = (kappa*xr + 16.0)/116.0;
	
	if(yr > epsilon)
	    fy = powf(yr, 1.0/3.0);
	else
	    fy = (kappa*yr + 16.0)/116.0;
	
	if(zr > epsilon)
	    fz = powf(zr, 1.0/3.0);
	else
	    fz = (kappa*zr + 16.0)/116.0;

//	__c_buffer->__l_vec[ index ] = ( float( 116.0*fy-16.0 ) / 100 ) * 255 ;
//	__c_buffer->__a_vec[ index ] = ( ( float( 500.0*(fx-fy) ) + 120 ) / 240 ) * 255 ;
//	__c_buffer->__b_vec[ index ] = ( ( float( 200.0*(fy-fz) ) + 120 ) / 240 ) * 255 ;
	
	__c_buffer->__lab[ 3*index + 0 ] = ( float( 116.0*fy-16.0 ) / 100 ) * 255 ;
	__c_buffer->__lab[ 3*index + 1 ] = ( ( float( 500.0*(fx-fy) ) + 120 ) / 240 ) * 255 ;
	__c_buffer->__lab[ 3*index + 2 ] = ( ( float( 200.0*(fy-fz) ) + 120 ) / 240 ) * 255 ;
	
	//if( index == 26894 )
	//    printf(" -- gpu -- (%i, %i, %i) = (%f, %f, %f) \n ", int( R[ index ] ), int( G[ index ] ), int( B[ index ] ), __c_buffer->__l_vec[ index ], __c_buffer->__a_vec[ index ], __c_buffer->__b_vec[ index ] );
	
}
