#ifndef _DMIAFNC_H_
#define _DMIAFNC_H_

#include "float3.h"

// Functions to not oversaturate the code in cuda/ext_AFNC_dmi.cu

// Returns intermediate value to simplify code
inline __device__ float mid1(float A1, float A2, float A3, float A4, float A5, float A6) {
    return (A5 - 0.5f*(A4*A6)/A2);
}

inline __device__ float mid2(float A1, float A2, float A3, float A4, float A5, float A6) {
    return (2.0f*A3*(1 - 0.25f*pow2(A6)/(A2*A3))); 
}

// Returns prefactor for DMI
inline __device__ float pref(float A1, float A2, float A3, float A4, float A5, float A6) {
    return (1.0f / (2.0f*A1*(1.0f - 0.25f*pow2(A4)/(A1*A2)) - pow2(mid1(A1,A2,A3,A4,A5,A6))/mid2(A1,A2,A3,A4,A5,A6))); 
}

inline __device__ float coef1(float A1, float A2, float A3, float A4, float A5, float A6) {
    return (0.5f*(A6*mid1(A1,A2,A3,A4,A5,A6))/(A2*mid2(A1,A2,A3,A4,A5,A6)) - 0.5f*A4/A2);
}

inline __device__ float coef2(float A1, float A2, float A3, float A4, float A5, float A6) {
    return (-mid1(A1,A2,A3,A4,A5,A6)/mid2(A1,A2,A3,A4,A5,A6));
}

// Returns mul * arr[i], or mul when arr == NULL;
//    return (arr == NULL)? (mul): (mul * arr[i]);

#endif
