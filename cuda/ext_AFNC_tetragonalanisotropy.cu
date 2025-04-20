#include "amul.h"
#include "float3.h"
#include <stdint.h>

// add cubic anisotropy field to B.
// B:      effective field in T
// m:      reduced magnetization (unit length)
// Ms:     saturation magnetization in A/m.
// K1:     Kc1 in J/m3
// K2:     Kc2 in T/m3
// C1, C2: anisotropy axes
//
// based on http://www.southampton.ac.uk/~fangohr/software/oxs_cubic8.html
extern "C" __global__ void
addtetragonalanisotropy(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
                    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                    float* __restrict__  Ms_, float  Ms_mul,
                    float* __restrict__  k2b_, float  k2b_mul,
                    float* __restrict__ c1x_, float c1x_mul,
                    float* __restrict__ c1y_, float c1y_mul,
                    float* __restrict__ c1z_, float c1z_mul,
                    float* __restrict__ c2x_, float c2x_mul,
                    float* __restrict__ c2y_, float c2y_mul,
                    float* __restrict__ c2z_, float c2z_mul,
                    int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float invMs = inv_Msat(Ms_, Ms_mul, i);
        float  k2b = amul(k2b_, k2b_mul, i) * invMs;
        float3 u3 = normalized(vmul(c1x_, c1y_, c1z_, c1x_mul, c1y_mul, c1z_mul, i));
        float3 u1 = normalized(vmul(c2x_, c2y_, c2z_, c2x_mul, c2y_mul, c2z_mul, i));
        float3 u2 = cross(u3,u1);
        float3 m  = make_float3(mx[i], my[i], mz[i]);

        float u1m = dot(u1, m);
        float u2m = dot(u2, m);

        float3 B = -2.0f*k2b*u1m*u2m*(u1m * u2 + u2m * u1);

        Bx[i] += B.x;
        By[i] += B.y;
        Bz[i] += B.z;
    }
}
