#include "amul.h"
#include "float3.h"
#include <stdint.h>

// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
extern "C" __global__ void
addExchangeAFNCCell(float* __restrict__ dst1x, float* __restrict__ dst1y, float* __restrict__ dst1z,
	float* __restrict__ dst2x, float* __restrict__ dst2y, float* __restrict__ dst2z,
	float* __restrict__ dst3x, float* __restrict__ dst3y, float* __restrict__ dst3z,
	float* __restrict__ m1x, float* __restrict__ m1y, float* __restrict__ m1z,
	float* __restrict__ m2x, float* __restrict__ m2y, float* __restrict__ m2z,
	float* __restrict__ m3x, float* __restrict__ m3y, float* __restrict__ m3z,
	float* __restrict__  Ms1_, float  Ms1_mul,
	float* __restrict__  Ms2_, float  Ms2_mul,
	float* __restrict__  Ms3_, float  Ms3_mul,
	float* __restrict__  Bex12_, float  Bex12_mul,
  float* __restrict__  Bex21_, float  Bex21_mul,
	float* __restrict__  Bex13_, float  Bex13_mul,
  float* __restrict__  Bex31_, float  Bex31_mul,
	float* __restrict__  Bex23_, float  Bex23_mul,
  float* __restrict__  Bex32_, float  Bex32_mul,
	int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float invMs1 = inv_Msat(Ms1_, Ms1_mul, i);
        float invMs2 = inv_Msat(Ms2_, Ms2_mul, i);
				float invMs3 = inv_Msat(Ms3_, Ms3_mul, i);
        float bex12 = amul(Bex12_, Bex12_mul, i);
        float bex21 = amul(Bex21_, Bex21_mul, i);
				float bex13 = amul(Bex13_, Bex13_mul, i);
        float bex31 = amul(Bex31_, Bex31_mul, i);
				float bex23 = amul(Bex23_, Bex23_mul, i);
        float bex32 = amul(Bex32_, Bex32_mul, i);

        dst1x[i] += invMs1*bex12*m2x[i] + invMs1*bex13*m3x[i];
        dst1y[i] += invMs1*bex12*m2y[i] + invMs1*bex13*m3y[i];
        dst1z[i] += invMs1*bex12*m2z[i] + invMs1*bex13*m3z[i];

        dst2x[i] += invMs2*bex21*m1x[i] + invMs2*bex23*m3x[i];
        dst2y[i] += invMs2*bex21*m1y[i] + invMs2*bex23*m3y[i];
        dst2z[i] += invMs2*bex21*m1z[i] + invMs2*bex23*m3z[i];

				dst3x[i] += invMs3*bex31*m1x[i] + invMs3*bex32*m2x[i];
				dst3y[i] += invMs3*bex31*m1y[i] + invMs3*bex32*m2y[i];
				dst3z[i] += invMs3*bex31*m1z[i] + invMs3*bex32*m2z[i];
    }
}
