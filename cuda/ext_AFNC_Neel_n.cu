#include "amul.h"
#include "float3.h"
#include <stdint.h>

extern "C" __global__ void

// normalize n: n=(m1*Ms1+m2*Ms2-m3*Ms3)/Ms0
NeelnAFNC(float* __restrict__ n0x, float* __restrict__ n0y, float* __restrict__ n0z,
	float* __restrict__ m1x, float* __restrict__ m1y, float* __restrict__ m1z,
	float* __restrict__ m2x, float* __restrict__ m2y, float* __restrict__ m2z,
	float* __restrict__ m3x, float* __restrict__ m3y, float* __restrict__ m3z,
	float* __restrict__  Ms1_, float  Ms1_mul,
	float* __restrict__  Ms2_, float  Ms2_mul,
	float* __restrict__  Ms3_, float  Ms3_mul,
	int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float  ms1 = amul(Ms1_, Ms1_mul, i);
        float  ms2 = amul(Ms2_, Ms2_mul, i);
				float  ms3 = amul(Ms3_, Ms3_mul, i);
        float invMs = 1.0f/(ms1+ms2+ms3);

				n0x[i] = (m1x[i]*ms1+m2x[i]*ms2-m3x[i]*ms3)*invMs;
        n0y[i] = (m1y[i]*ms1+m2y[i]*ms2-m3y[i]*ms3)*invMs;
        n0z[i] = (m1z[i]*ms1+m2z[i]*ms2-m3z[i]*ms3)*invMs;

			}
	}
