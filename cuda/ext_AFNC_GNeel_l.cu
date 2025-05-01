#include "amul.h"
#include "float3.h"
#include <stdint.h>

extern "C" __global__ void

// normalize l: l=(m1*Ms1-m2*Ms2+m3*Ms3)/Ms0
GNeellAFNC(float* __restrict__ l0x, float* __restrict__ l0y, float* __restrict__ l0z,
	float* __restrict__ m1x, float* __restrict__ m1y, float* __restrict__ m1z,
	float* __restrict__ m2x, float* __restrict__ m2y, float* __restrict__ m2z,
	float* __restrict__ m3x, float* __restrict__ m3y, float* __restrict__ m3z,
	float* __restrict__  Ms1_, float  Ms1_mul,
	float* __restrict__  Ms2_, float  Ms2_mul,
	float* __restrict__  Ms3_, float  Ms3_mul,
	float g1, float g2, float g3,
	int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float  ms1 = amul(Ms1_, Ms1_mul, i)/g1;
        float  ms2 = amul(Ms2_, Ms2_mul, i)/g2;
				float  ms3 = amul(Ms3_, Ms3_mul, i)/g3;
        float invMs = 1.0f/(ms1+ms2+ms3);

				l0x[i] = (m1x[i]*ms1-m2x[i]*ms2+m3x[i]*ms3)*invMs;
        l0y[i] = (m1y[i]*ms1-m2y[i]*ms2+m3y[i]*ms3)*invMs;
        l0z[i] = (m1z[i]*ms1-m2z[i]*ms2+m3z[i]*ms3)*invMs;

			}
	}
