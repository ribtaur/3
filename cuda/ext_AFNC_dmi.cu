#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "dmiafnc.h"

// Exchange + Dzyaloshinskii-Moriya interaction according to
// Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// Taking into account proper boundary conditions.
// m: normalized magnetization
// H: effective field in Tesla
// D: dmi strength / Msat, in Tesla*m
// A: Aex/Msat

// New Notation;
// H1x...,H1z: output field for the current lattice (1 or 2)
// m1x,m1y,m1z m2x,m2y,m2z input m components of sublattices
// Ms1, Ms2 saturation magnetizations (probably Ms2 Nor needed, to remove if possible)
// aLUT2d1,dLUT2d1,aLUT2d2,dLUT2d2,aLUT2d12,bex12,bex21 input matrix of exchanges named in the following as:

// Aex1, Dind1, Aex2, Dind2, Aex3, Dind3, Aexll12, Aexll13, Aexll23
// m10, m20, m30 central cell of each lattice
// m1v, m2v, m3v neighbour for each cell in any direction
// Exchange constants are named in the same way in all the steps

// Removed intracell exchange (and references to Bex's constants), an external function already considers it

extern "C" __global__ void
adddmiAFNC(float *__restrict__ dst1x, float *__restrict__ dst1y, float *__restrict__ dst1z,
	float *__restrict__ dst2x, float *__restrict__ dst2y, float *__restrict__ dst2z,
	float *__restrict__ dst3x, float *__restrict__ dst3y, float *__restrict__ dst3z,
 	float *__restrict__ m1x, float *__restrict__ m1y, float *__restrict__ m1z,
	float *__restrict__ m2x, float *__restrict__ m2y, float *__restrict__ m2z,
	float *__restrict__ m3x, float *__restrict__ m3y, float *__restrict__ m3z,
 	float *__restrict__ Ms1_, float Ms1_mul,
	float *__restrict__ Ms2_, float Ms2_mul,
	float *__restrict__ Ms3_, float Ms3_mul,
	float *__restrict__ aLUT2d1, float *__restrict__ dLUT2d1,
	float *__restrict__ aLUT2d2, float *__restrict__ dLUT2d2,
	float *__restrict__ aLUT2d3, float *__restrict__ dLUT2d3,
	float *__restrict__ aLUT2d12, float *__restrict__ aLUT2d13, float *__restrict__ aLUT2d23,
 	uint8_t * __restrict__ regions, float cx, float cy, float cz,
 	int Nx, int Ny, int Nz, uint8_t PBC, uint8_t OpenBC)
{

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

	int I = idx(ix, iy, iz);	// central cell index
	float3 h1 = make_float3(0.0, 0.0, 0.0);	// add to Hex1
	float3 h2 = make_float3(0.0, 0.0, 0.0);	// add to Hex2
	float3 h3 = make_float3(0.0, 0.0, 0.0);	// add to Hex3
	float3 m10 = make_float3(m1x[I], m1y[I], m1z[I]);	// central m1
	float3 m20 = make_float3(m2x[I], m2y[I], m2z[I]);	// central m2
	float3 m30 = make_float3(m3x[I], m3y[I], m3z[I]);	// central m3
	uint8_t r0 = regions[I];
	int i_;			// neighbor index

	if (is0(m10) && is0(m20) && is0(m30)) {
		return;
	}

	{
		// x derivatives (along length)

		float3 m1v = make_float3(0.0f, 0.0f, 0.0f);	// left neighbor
		float3 m2v = make_float3(0.0f, 0.0f, 0.0f);	// left neighbor
		float3 m3v = make_float3(0.0f, 0.0f, 0.0f);	// left neighbor
		i_ = idx(lclampx(ix - 1), iy, iz);	// load neighbor m if inside grid, keep 0 otherwise
		if (ix - 1 >= 0 || PBCx) {
			m1v = make_float3(m1x[i_], m1y[i_], m1z[i_]);
			m2v = make_float3(m2x[i_], m2y[i_], m2z[i_]);
			m3v = make_float3(m3x[i_], m3y[i_], m3z[i_]);
		}
		// introducing 2 regions neighbours for ferro/ferri simulations (m1 or m2 zero in different parts)
		// Note that r13 and r23 are defined the same way, it just checks if m3v
		// from neighbor is zero, as m20 and m10 are nonzero.
		uint8_t r11 = is0(m1v) ? r0 : regions[i_];	// don't use inter region params if m1v=0
		uint8_t r12 = is0(m2v) ? r0 : regions[i_];	// don't use inter region params if m2v=0
		uint8_t r13 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0
		uint8_t r23 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0

		float Aex1 = aLUT2d1[symidx(r0, r11)];	// inter-region Aex1
		float Dind1 = dLUT2d1[symidx(r0, r11)];	// inter-region Dex1

		float Aex2 = aLUT2d2[symidx(r0, r12)];	// inter-region Aex2
		float Dind2 = dLUT2d2[symidx(r0, r12)];	// inter-region Dex2

		float Aex3 = aLUT2d2[symidx(r0, r13)];	// inter-region Aex3
		float Dind3 = dLUT2d2[symidx(r0, r13)];	// inter-region Dex3

		float Aexll12 = aLUT2d12[symidx(r0, r12)];	// inter-region Aexll12
		float Aexll13 = aLUT2d13[symidx(r0, r13)];	// inter-region Aexll13
		float Aexll23 = aLUT2d23[symidx(r0, r23)];	// inter-region Aexll23

/*		if (blockIdx.x == 0 && threadIdx.x == 0) {

		printf("mid1: %f\n", mid1(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23));
		printf("mid2: %f\n", mid2(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23));

		printf("pref: %f\n", pref(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23));
		printf("coef1: %f\n", coef1(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23));
		printf("coef2: %f\n", coef2(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23));
		}
*/
		// Left neighbor (-x)

		if (!is0(m1v) || !OpenBC) {	// do nothing at an open boundary
			if (is0(m1v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);
				float Coef1 = coef1(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);
				float Coef2 = coef2(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);

				m1v.x = m10.x - Pref * (-cx) * (Dind1 * m10.z + Coef1 * Dind2 * m20.z + Coef2 * Dind3 * m30.z);
				m1v.y = m10.y;
				m1v.z = m10.z + Pref * (-cx) * (Dind1 * m10.x + Coef1 * Dind2 * m20.x + Coef2 * Dind3 * m30.x);
			}
			if (is0(m2v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);
				float Coef1 = coef1(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);
				float Coef2 = coef2(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);

				m2v.x = m20.x - Pref * (-cx) * (Coef1 * Dind1 * m10.z + Dind2 * m20.z + Coef2 * Dind3 * m30.z);
				m2v.y = m20.y;
				m2v.z = m20.z + Pref * (-cx) * (Coef1 * Dind1 * m10.x + Dind2 * m20.x + Coef2 * Dind3 * m30.x);
			}
			if (is0(m3v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);
				float Coef1 = coef1(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);
				float Coef2 = coef2(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);

				m3v.x = m30.x - Pref * (-cx) * (Coef1 * Dind1 * m10.z + Coef2 * Dind2 * m20.z + Dind3 * m30.z);
				m3v.y = m30.y;
				m3v.z = m30.z + Pref * (-cx) * (Coef1 * Dind1 * m10.x + Coef2 * Dind2 * m20.x + Dind3 * m30.x);
			}
			// Sublattice 1
			// Echange by Aex1
			h1 += (2.0f * Aex1 / (cx * cx)) * (m1v - m10);	// Usual exchange
			h1.x += (Dind1 / cx) * (-m1v.z);	// DMI
			h1.z -= (Dind1 / cx) * (-m1v.x);	// DMI
			// Echange by Aexll
			h1 += Aexll12 / (cx * cx) * (m2v - m20);	// Exchange between lattices 1-2
			h1 += Aexll13 / (cx * cx) * (m3v - m30);	// Exchange between lattices 1-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h1 += Bex12 * m20; Already considered

			// Sublattice 2
			// Echange by Aex2
			h2 += (2.0f * Aex2 / (cx * cx)) * (m2v - m20);	// Usual exchange
			h2.x += (Dind2 / cx) * (-m2v.z);	// DMI
			h2.z -= (Dind2 / cx) * (-m2v.x);	// DMI
			// Echange by Aexll
			h2 += Aexll12 / (cx * cx) * (m1v - m10);	// Exchange between lattices 1-2
			h2 += Aexll23 / (cx * cx) * (m3v - m30);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10; Already considered

			// Sublattice 3
			// Echange by Aex3
			h3 += (2.0f * Aex3 / (cx * cx)) * (m3v - m30);	// Usual exchange
			h3.x += (Dind3 / cx) * (-m3v.z);	// DMI
			h3.z -= (Dind3 / cx) * (-m3v.x);	// DMI
			// Echange by Aexll
			h3 += Aexll13 / (cx * cx) * (m1v - m10);	// Exchange between lattices 1-3
			h3 += Aexll23 / (cx * cx) * (m2v - m20);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10; Already considered
		}
	}

	{
		float3 m1v = make_float3(0.0f, 0.0f, 0.0f);	// right neighbor
		float3 m2v = make_float3(0.0f, 0.0f, 0.0f);	// right neighbor
		float3 m3v = make_float3(0.0f, 0.0f, 0.0f);	// right neighbor
		i_ = idx(hclampx(ix + 1), iy, iz);	// load neighbor m if inside grid, keep 0 otherwise
		if (ix + 1 < Nx || PBCx) {
			m1v = make_float3(m1x[i_], m1y[i_], m1z[i_]);
			m2v = make_float3(m2x[i_], m2y[i_], m2z[i_]);
			m3v = make_float3(m3x[i_], m3y[i_], m3z[i_]);
		}
		// introducing 2 regions neighbours for ferro/ferri simulations (m1 or m2 zero in different parts)
		// Note that r13 and r23 are defined the same way, it just checks if m3v
		// from neighbor is zero, as m20 and m10 are nonzero.
		uint8_t r11 = is0(m1v) ? r0 : regions[i_];	// don't use inter region params if m1v=0
		uint8_t r12 = is0(m2v) ? r0 : regions[i_];	// don't use inter region params if m2v=0
		uint8_t r13 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0
		uint8_t r23 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0

		float Aex1 = aLUT2d1[symidx(r0, r11)];	// inter-region Aex1
		float Dind1 = dLUT2d1[symidx(r0, r11)];	// inter-region Dex1

		float Aex2 = aLUT2d2[symidx(r0, r12)];	// inter-region Aex2
		float Dind2 = dLUT2d2[symidx(r0, r12)];	// inter-region Dex2

		float Aex3 = aLUT2d2[symidx(r0, r13)];	// inter-region Aex3
		float Dind3 = dLUT2d2[symidx(r0, r13)];	// inter-region Dex3

		float Aexll12 = aLUT2d12[symidx(r0, r12)];	// inter-region Aexll12
		float Aexll13 = aLUT2d13[symidx(r0, r13)];	// inter-region Aexll13
		float Aexll23 = aLUT2d23[symidx(r0, r23)];	// inter-region Aexll23

		// Right neighbor (+x)

		if (!is0(m1v) || !OpenBC) {	// do nothing at an open boundary
			if (is0(m1v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);
				float Coef1 = coef1(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);
				float Coef2 = coef2(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);

				m1v.x = m10.x - Pref * (cx) * (Dind1 * m10.z + Coef1 * Dind2 * m20.z + Coef2 * Dind3 * m30.z);
				m1v.y = m10.y;
				m1v.z = m10.z + Pref * (cx) * (Dind1 * m10.x + Coef1 * Dind2 * m20.x + Coef2 * Dind3 * m30.x);
			}
			if (is0(m2v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);
				float Coef1 = coef1(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);
				float Coef2 = coef2(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);

				m2v.x = m20.x - Pref * (cx) * (Coef1 * Dind1 * m10.z + Dind2 * m20.z + Coef2 * Dind3 * m30.z);
				m2v.y = m20.y;
				m2v.z = m20.z + Pref * (cx) * (Coef1 * Dind1 * m10.x + Dind2 * m20.x + Coef2 * Dind3 * m30.x);
			}
			if (is0(m3v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);
				float Coef1 = coef1(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);
				float Coef2 = coef2(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);

				m3v.x = m30.x - Pref * (cx) * (Coef1 * Dind1 * m10.z + Coef2 * Dind2 * m20.z + Dind3 * m30.z);
				m3v.y = m30.y;
				m3v.z = m30.z + Pref * (cx) * (Coef1 * Dind1 * m10.x + Coef2 * Dind2 * m20.x + Dind3 * m30.x);
			}

			// Sublattice 1
			// Echange by Aex1
			h1 += (2.0f * Aex1 / (cx * cx)) * (m1v - m10);	// Usual exchange
			h1.x += (Dind1 / cx) * (m1v.z);	// DMI
			h1.z -= (Dind1 / cx) * (m1v.x);	// DMI
			// Echange by Aexll
			h1 += Aexll12 / (cx * cx) * (m2v - m20);	// Exchange between lattices 1-2
			h1 += Aexll13 / (cx * cx) * (m3v - m30);	// Exchange between lattices 1-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h1 += Bex12 * m20;

			// Sublattice 2
			// Echange by Aex2
			h2 += (2.0f * Aex2 / (cx * cx)) * (m2v - m20);	// Usual exchange
			h2.x += (Dind2 / cx) * (m2v.z);	// DMI
			h2.z -= (Dind2 / cx) * (m2v.x);	// DMI
			// Echange by Aexll
			h2 += Aexll12 / (cx * cx) * (m1v - m10);	// Exchange between lattices 1-2
			h2 += Aexll23 / (cx * cx) * (m3v - m30);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;

			// Sublattice 3
			// Echange by Aex3
			h3 += (2.0f * Aex3 / (cx * cx)) * (m3v - m30);	// Usual exchange
			h3.x += (Dind3 / cx) * (m3v.z);	// DMI
			h3.z -= (Dind3 / cx) * (m3v.x);	// DMI
			// Echange by Aexll
			h3 += Aexll13 / (cx * cx) * (m1v - m10);	// Exchange between lattices 1-3
			h3 += Aexll23 / (cx * cx) * (m2v - m20);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;
		}
	}

	// y derivatives (along height)
	{
		float3 m1v = make_float3(0.0f, 0.0f, 0.0f);	// lower neighbor
		float3 m2v = make_float3(0.0f, 0.0f, 0.0f);	// lower neighbor
		float3 m3v = make_float3(0.0f, 0.0f, 0.0f);	// lower neighbor
		i_ = idx(ix, lclampy(iy - 1), iz);	// load neighbor m if inside grid, keep 0 otherwise
		if (iy - 1 >= 0 || PBCy) {
			m1v = make_float3(m1x[i_], m1y[i_], m1z[i_]);
			m2v = make_float3(m2x[i_], m2y[i_], m2z[i_]);
			m3v = make_float3(m3x[i_], m3y[i_], m3z[i_]);
		}

		// introducing 2 regions neighbours for ferro/ferri simulations (m1 or m2 zero in different parts)
		// Note that r13 and r23 are defined the same way, it just checks if m3v
		// from neighbor is zero, as m20 and m10 are nonzero.
		uint8_t r11 = is0(m1v) ? r0 : regions[i_];	// don't use inter region params if m1v=0
		uint8_t r12 = is0(m2v) ? r0 : regions[i_];	// don't use inter region params if m2v=0
		uint8_t r13 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0
		uint8_t r23 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0

		float Aex1 = aLUT2d1[symidx(r0, r11)];	// inter-region Aex1
		float Dind1 = dLUT2d1[symidx(r0, r11)];	// inter-region Dex1

		float Aex2 = aLUT2d2[symidx(r0, r12)];	// inter-region Aex2
		float Dind2 = dLUT2d2[symidx(r0, r12)];	// inter-region Dex2

		float Aex3 = aLUT2d2[symidx(r0, r13)];	// inter-region Aex3
		float Dind3 = dLUT2d2[symidx(r0, r13)];	// inter-region Dex3

		float Aexll12 = aLUT2d12[symidx(r0, r12)];	// inter-region Aexll12
		float Aexll13 = aLUT2d13[symidx(r0, r13)];	// inter-region Aexll13
		float Aexll23 = aLUT2d23[symidx(r0, r23)];	// inter-region Aexll23

		// Front neighbor (-y)

		if (!is0(m1v) || !OpenBC) {	// do nothing at an open boundary
			if (is0(m1v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);
				float Coef1 = coef1(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);
				float Coef2 = coef2(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);

				m1v.x = m10.x;
				m1v.y = m10.y - Pref * (-cy) * (Dind1 * m10.z + Coef1 * Dind2 * m20.z + Coef2 * Dind3 * m30.z);
				m1v.z = m10.z + Pref * (-cy) * (Dind1 * m10.y + Coef1 * Dind2 * m20.y + Coef2 * Dind3 * m30.y);
			}
			if (is0(m2v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);
				float Coef1 = coef1(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);
				float Coef2 = coef2(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);

				m2v.x = m20.x;
				m2v.y = m20.y - Pref * (-cy) * (Coef1 * Dind1 * m10.z + Dind2 * m20.z + Coef2 * Dind3 * m30.z);
				m2v.z = m20.z + Pref * (-cy) * (Coef1 * Dind1 * m10.y + Dind2 * m20.y + Coef2 * Dind3 * m30.y);
			}
			if (is0(m3v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);
				float Coef1 = coef1(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);
				float Coef2 = coef2(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);

				m3v.x = m30.x;
				m3v.y = m30.y - Pref * (-cy) * (Coef1 * Dind1 * m10.z + Coef2 * Dind2 * m20.z + Dind3 * m30.z);
				m3v.z = m30.z + Pref * (-cy) * (Coef1 * Dind1 * m10.y + Coef2 * Dind2 * m20.y + Dind3 * m30.y);
			}

			// Sublattice 1
			// Echange by Aex1
			h1 += (2.0f * Aex1 / (cy * cy)) * (m1v - m10);	// Usual exchange
			h1.y += (Dind1 / cy) * (-m1v.z);	// DMI
			h1.z -= (Dind1 / cy) * (-m1v.y);	// DMI
			// Echange by Aexll
			h1 += Aexll12 / (cy * cy) * (m2v - m20);	// Exchange between lattices 1-2
			h1 += Aexll13 / (cy * cy) * (m3v - m30);	// Exchange between lattices 1-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h1 += Bex12 * m20;

			// Sublattice 2
			// Echange by Aex2
			h2 += (2.0f * Aex2 / (cy * cy)) * (m2v - m20);	// Usual exchange
			h2.y += (Dind2 / cy) * (-m2v.z);	// DMI
			h2.z -= (Dind2 / cy) * (-m2v.y);	// DMI
			// Echange by Aexll
			h2 += Aexll12 / (cy * cy) * (m1v - m10);	// Exchange between lattices 1-2
			h2 += Aexll23 / (cy * cy) * (m3v - m30);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;

			// Sublattice 3
			// Echange by Aex3
			h3 += (2.0f * Aex3 / (cy * cy)) * (m3v - m30);	// Usual exchange
			h3.y += (Dind3 / cy) * (-m3v.z);	// DMI
			h3.z -= (Dind3 / cy) * (-m3v.y);	// DMI
			// Echange by Aexll
			h3 += Aexll13 / (cy * cy) * (m1v - m10);	// Exchange between lattices 1-3
			h3 += Aexll23 / (cy * cy) * (m2v - m20);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;
		}
	}

	{
		float3 m1v = make_float3(0.0f, 0.0f, 0.0f);	// upper neighbor
		float3 m2v = make_float3(0.0f, 0.0f, 0.0f);	// upper neighbor
		float3 m3v = make_float3(0.0f, 0.0f, 0.0f);	// upper neighbor
		i_ = idx(ix, hclampy(iy + 1), iz);
		if (iy + 1 < Ny || PBCy) {
			m1v = make_float3(m1x[i_], m1y[i_], m1z[i_]);
			m2v = make_float3(m2x[i_], m2y[i_], m2z[i_]);
			m3v = make_float3(m3x[i_], m3y[i_], m3z[i_]);
		}

		// introducing 2 regions neighbours for ferro/ferri simulations (m1 or m2 zero in different parts)
		// Note that r13 and r23 are defined the same way, it just checks if m3v
		// from neighbor is zero, as m20 and m10 are nonzero.
		uint8_t r11 = is0(m1v) ? r0 : regions[i_];	// don't use inter region params if m1v=0
		uint8_t r12 = is0(m2v) ? r0 : regions[i_];	// don't use inter region params if m2v=0
		uint8_t r13 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0
		uint8_t r23 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0

		float Aex1 = aLUT2d1[symidx(r0, r11)];	// inter-region Aex1
		float Dind1 = dLUT2d1[symidx(r0, r11)];	// inter-region Dex1

		float Aex2 = aLUT2d2[symidx(r0, r12)];	// inter-region Aex2
		float Dind2 = dLUT2d2[symidx(r0, r12)];	// inter-region Dex2

		float Aex3 = aLUT2d2[symidx(r0, r13)];	// inter-region Aex3
		float Dind3 = dLUT2d2[symidx(r0, r13)];	// inter-region Dex3

		float Aexll12 = aLUT2d12[symidx(r0, r12)];	// inter-region Aexll12
		float Aexll13 = aLUT2d13[symidx(r0, r13)];	// inter-region Aexll13
		float Aexll23 = aLUT2d23[symidx(r0, r23)];	// inter-region Aexll23

		// Back neighbor (+y)

		if (!is0(m1v) || !OpenBC) {	// do nothing at an open boundary
			if (is0(m1v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);
				float Coef1 = coef1(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);
				float Coef2 = coef2(Aex1, Aex2, Aex3, Aexll12, Aexll13, Aexll23);

				m1v.x = m10.x;
				m1v.y = m10.y - Pref * (cy) * (Dind1 * m10.z + Coef1 * Dind2 * m20.z + Coef2 * Dind3 * m30.z);
				m1v.z = m10.z + Pref * (cy) * (Dind1 * m10.y + Coef1 * Dind2 * m20.y + Coef2 * Dind3 * m30.y);
			}
			if (is0(m2v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);
				float Coef1 = coef1(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);
				float Coef2 = coef2(Aex2, Aex1, Aex3, Aexll12, Aexll23, Aexll13);

				m2v.x = m20.x;
				m2v.y = m20.y - Pref * (cy) * (Coef1 * Dind1 * m10.z + Dind2 * m20.z + Coef2 * Dind3 * m30.z);
				m2v.z = m20.z + Pref * (cy) * (Coef1 * Dind1 * m10.y + Dind2 * m20.y + Coef2 * Dind3 * m30.y);
			}
			if (is0(m3v)) {	// neighbor missing, use BC
				float Pref  = pref(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);
				float Coef1 = coef1(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);
				float Coef2 = coef2(Aex3, Aex1, Aex2, Aexll13, Aexll23, Aexll12);

				m3v.x = m30.x;
				m3v.y = m30.y - Pref * (cy) * (Coef1 * Dind1 * m10.z + Coef2 * Dind2 * m20.z + Dind3 * m30.z);
				m3v.z = m30.z + Pref * (cy) * (Coef1 * Dind1 * m10.y + Coef2 * Dind2 * m20.y + Dind3 * m30.y);
			}

			// Sublattice 1
			// Echange by Aex1
			h1 += (2.0f * Aex1 / (cy * cy)) * (m1v - m10);	// Usual exchange
			h1.y += (Dind1 / cy) * (m1v.z);	// DMI
			h1.z -= (Dind1 / cx) * (m1v.y);	// DMI
			// Echange by Aexll
			h1 += Aexll12 / (cy * cy) * (m2v - m20);	// Exchange between lattices 1-2
			h1 += Aexll13 / (cy * cy) * (m3v - m30);	// Exchange between lattices 1-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h1 += Bex12 * m20;

			// Sublattice 2
			// Echange by Aex2
			h2 += (2.0f * Aex2 / (cy * cy)) * (m2v - m20);	// Usual exchange
			h2.y += (Dind2 / cy) * (m2v.z);	// DMI
			h2.z -= (Dind2 / cx) * (m2v.y);	// DMI
			// Echange by Aexll
			h2 += Aexll12 / (cy * cy) * (m1v - m10);	// Exchange between lattices 1-2
			h2 += Aexll23 / (cy * cy) * (m3v - m30);	// Exchange between lattices 1-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;

			// Sublattice 3
			// Echange by Aex3
			h3 += (2.0f * Aex3 / (cy * cy)) * (m3v - m30);	// Usual exchange
			h3.y += (Dind3 / cy) * (m3v.z);	// DMI
			h3.z -= (Dind3 / cx) * (m3v.y);	// DMI
			// Echange by Aexll
			h3 += Aexll13 / (cy * cy) * (m1v - m10);	// Exchange between lattices 1-3
			h3 += Aexll23 / (cy * cy) * (m2v - m20);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;
		}
	}

	// only take vertical derivative for 3D sim
	if (Nz != 1) {
		// bottom neighbor
		{
			i_ = idx(ix, iy, lclampz(iz - 1));
			float3 m1v = make_float3(m1x[i_], m1y[i_], m1z[i_]);
			m1v = (is0(m1v) ? m10 : m1v);	// Neumann BC
			float3 m2v = make_float3(m2x[i_], m2y[i_], m2z[i_]);
			m2v = (is0(m2v) ? m20 : m2v);
			float3 m3v = make_float3(m3x[i_], m3y[i_], m3z[i_]);
			m3v = (is0(m3v) ? m30 : m3v);

			// introducing 2 regions neighbours for ferro/ferri simulations (m1 or m2 zero in different parts)
			// Note that r13 and r23 are defined the same way, it just checks if m3v
			// from neighbor is zero, as m20 and m10 are nonzero.
			uint8_t r11 = is0(m1v) ? r0 : regions[i_];	// don't use inter region params if m1v=0
			uint8_t r12 = is0(m2v) ? r0 : regions[i_];	// don't use inter region params if m2v=0
			uint8_t r13 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0
			uint8_t r23 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0

			float Aex1 = aLUT2d1[symidx(r0, r11)];	// inter-region Aex1
			float Aex2 = aLUT2d2[symidx(r0, r12)];	// inter-region Aex2
			float Aex3 = aLUT2d2[symidx(r0, r13)];	// inter-region Aex3
			float Aexll12 = aLUT2d12[symidx(r0, r12)];	// inter-region Aexll12
			float Aexll13 = aLUT2d13[symidx(r0, r13)];	// inter-region Aexll13
			float Aexll23 = aLUT2d23[symidx(r0, r23)];	// inter-region Aexll23

			// Sublattice 1
			h1 += (2.0f * Aex1 / (cz * cz)) * (m1v - m10);	// Exchange only
			// Echange by Aexll
			h1 += Aexll12 / (cz * cz) * (m2v - m20);	// Exchange between lattices 1-2
			h1 += Aexll13 / (cz * cz) * (m3v - m30);	// Exchange between lattices 1-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h1 += Bex12 * m20;

			// Sublattice 2
			h2 += (2.0f * Aex2 / (cz * cz)) * (m2v - m20);	// Exchange only
			// Echange by Aexll
			h2 += Aexll12 / (cz * cz) * (m1v - m10);	// Exchange between lattices 1-2
			h2 += Aexll23 / (cz * cz) * (m3v - m30);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;

			// Sublattice 3
			h3 += (2.0f * Aex3 / (cz * cz)) * (m3v - m30);	// Exchange only
			// Echange by Aexll
			h3 += Aexll13 / (cz * cz) * (m1v - m10);	// Exchange between lattices 1-3
			h3 += Aexll23 / (cz * cz) * (m2v - m20);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;
		}

		// top neighbor
		{
			i_ = idx(ix, iy, hclampz(iz + 1));
			float3 m1v = make_float3(m1x[i_], m1y[i_], m1z[i_]);
			m1v = (is0(m1v) ? m10 : m1v);	// Neumann BC
			float3 m2v = make_float3(m2x[i_], m2y[i_], m2z[i_]);
			m2v = (is0(m2v) ? m20 : m2v);
			float3 m3v = make_float3(m3x[i_], m3y[i_], m3z[i_]);
			m3v = (is0(m3v) ? m30 : m3v);

			// introducing 2 regions neighbours for ferro/ferri simulations (m1 or m2 zero in different parts)
			// Note that r13 and r23 are defined the same way, it just checks if m3v
			// from neighbor is zero, as m20 and m10 are nonzero.
			uint8_t r11 = is0(m1v) ? r0 : regions[i_];	// don't use inter region params if m1v=0
			uint8_t r12 = is0(m2v) ? r0 : regions[i_];	// don't use inter region params if m2v=0
			uint8_t r13 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0
			uint8_t r23 = is0(m3v) ? r0 : regions[i_];	// don't use inter region params if m3v=0

			float Aex1 = aLUT2d1[symidx(r0, r11)];	// inter-region Aex1
			float Aex2 = aLUT2d2[symidx(r0, r12)];	// inter-region Aex2
			float Aex3 = aLUT2d2[symidx(r0, r13)];	// inter-region Aex3
			float Aexll12 = aLUT2d12[symidx(r0, r12)];	// inter-region Aexll12
			float Aexll13 = aLUT2d13[symidx(r0, r13)];	// inter-region Aexll13
			float Aexll23 = aLUT2d23[symidx(r0, r23)];	// inter-region Aexll23

			// Sublattice 1
			h1 += (2.0f * Aex1 / (cz * cz)) * (m1v - m10);	// Exchange only
			// Echange by Aexll
			h1 += Aexll12 / (cz * cz) * (m2v - m20);	// Exchange between lattices 1-2
			h1 += Aexll13 / (cz * cz) * (m3v - m30);	// Exchange between lattices 1-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h1 += Bex12 * m20;

			// Sublattice 2
			h2 += (2.0f * Aex2 / (cz * cz)) * (m2v - m20);	// Exchange only
			// Echange by Aexll
			h2 += Aexll12 / (cz * cz) * (m1v - m10);	// Exchange between lattices 1-2
			h2 += Aexll23 / (cz * cz) * (m3v - m30);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;

			// Sublattice 3
			h3 += (2.0f * Aex3 / (cz * cz)) * (m3v - m30);	// Exchange only
			// Echange by Aexll
			h3 += Aexll13 / (cz * cz) * (m1v - m10);	// Exchange between lattices 1-3
			h3 += Aexll23 / (cz * cz) * (m2v - m20);	// Exchange between lattices 2-3
			// Intracell exchange (between sublatices, local, no derivatives)
			// h2 += Bex21 * m10;
		}
	}

	// write back, result is H + Hdmi + Hex
	float invMs1 = inv_Msat(Ms1_, Ms1_mul, I);
	dst1x[I] += h1.x * invMs1;
	dst1y[I] += h1.y * invMs1;
	dst1z[I] += h1.z * invMs1;
	float invMs2 = inv_Msat(Ms2_, Ms2_mul, I);
	dst2x[I] += h2.x * invMs2;
	dst2y[I] += h2.y * invMs2;
	dst2z[I] += h2.z * invMs2;
	float invMs3 = inv_Msat(Ms3_, Ms3_mul, I);
	dst3x[I] += h3.x * invMs3;
	dst3y[I] += h3.y * invMs3;
	dst3z[I] += h3.z * invMs3;
}

// Note on boundary conditions.
//
// We need the derivative and laplacian of m in point A, but e.g. C lies out of the boundaries.
// We use the boundary condition in B (derivative of the magnetization) to extrapolate m to point C:
//      m_C = m_A + (dm/dx)|_B * cellsize
//
// When point C is inside the boundary, we just use its actual value.
//
// Then we can take the central derivative in A:
//      (dm/dx)|_A = (m_C - m_D) / (2*cellsize)
// And the laplacian:
//      lapl(m)|_A = (m_C + m_D - 2*m_A) / (cellsize^2)
//
// All these operations should be second order as they involve only central derivatives.
//
//    ------------------------------------------------------------------ *
//   |                                                   |             C |
//   |                                                   |          **   |
//   |                                                   |        ***    |
//   |                                                   |     ***       |
//   |                                                   |   ***         |
//   |                                                   | ***           |
//   |                                                   B               |
//   |                                               *** |               |
//   |                                            ***    |               |
//   |                                         ****      |               |
//   |                                     ****          |               |
//   |                                  ****             |               |
//   |                              ** A                 |               |
//   |                         *****                     |               |
//   |                   ******                          |               |
//   |          *********                                |               |
//   |D ********                                         |               |
//   |                                                   |               |
//   +----------------+----------------+-----------------+---------------+
//  -1              -0.5               0               0.5               1
//                                 x
