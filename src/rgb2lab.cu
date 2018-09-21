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
	float _b = (float)RGB[ index * 3 + 0 ] * 0.0039216f;
	float _g = (float)RGB[ index * 3 + 1 ] * 0.0039216f;
	float _r = (float)RGB[ index * 3 + 2 ] * 0.0039216f;

	float x = _r*0.412453f + _g*0.357580f + _b*0.180423f;
	float y = _r*0.212671f + _g*0.715160f + _b*0.072169f;
	float z = _r*0.019334f + _g*0.119193f + _b*0.950227f;

	float epsilon = 0.008856f;	//actual CIE standard

	float Xr = 0.950456f;	//reference white
	float Yr = 1.0f;		//reference white
	float Zr = 1.088754f;	//reference white

	float xr = x / Xr;
	float yr = y / Yr;
	float zr = z / Zr;

	float fx, fy, fz;
	if (xr > epsilon)	fx = pow(xr, 1.0f / 3.0f);
	else				fx = 7.787f*xr + 0.137931034f;
	
	if (yr > epsilon)	fy = pow(yr, 1.0f / 3.0f);
	else				fy = 7.787f*yr + 0.137931034f;
	
	if (zr > epsilon)	fz = pow(zr, 1.0f / 3.0f);
	else				fz = 7.787f*zr + 0.137931034f;

	__c_buffer->__lab[ 3*index + 0 ] = ( 116.0f*fy - 16.0f ) * 2.55f;
	__c_buffer->__lab[ 3*index + 1 ] = ( 500.0f*(fx - fy) ) + 128;
	__c_buffer->__lab[ 3*index + 2 ] = ( 200.0f*(fy - fz) ) + 128;
	
	//if( index == 26894 )
	//    printf(" -- gpu -- (%i, %i, %i) = (%f, %f, %f) \n ", int( R[ index ] ), int( G[ index ] ), int( B[ index ] ), __c_buffer->__l_vec[ index ], __c_buffer->__a_vec[ index ], __c_buffer->__b_vec[ index ] );
	
}
