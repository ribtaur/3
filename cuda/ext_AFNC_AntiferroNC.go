package cuda

import (
	"github.com/mumax/3/data"
	//	"github.com/mumax/3/util"
	"unsafe"
)

func NormalizeAFNC(m0, m1, m2, m3 *data.Slice, Ms0, Ms1, Ms2, Ms3 MSlice) {
	N := m0.Len()
	cfg := make1DConf(N)
	k_normalizeAFNC_async(m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Ms0.DevPtr(0), Ms0.Mul(0),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		Ms3.DevPtr(0), Ms3.Mul(0), N, cfg)

}

func AddExchangeAFNCCell(dst1, dst2, dst3, m1, m2, m3 *data.Slice, Ms1, Ms2, Ms3, bex12, bex21, bex13, bex31, bex23, bex32 MSlice) {
	N := m1.Len()
	cfg := make1DConf(N)
	k_addExchangeAFNCCell_async(dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
		dst2.DevPtr(X), dst2.DevPtr(Y), dst2.DevPtr(Z),
		dst3.DevPtr(X), dst3.DevPtr(Y), dst3.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		Ms3.DevPtr(0), Ms3.Mul(0),
		bex12.DevPtr(0), bex12.Mul(0),
		bex21.DevPtr(0), bex21.Mul(0),
		bex13.DevPtr(0), bex13.Mul(0),
		bex31.DevPtr(0), bex31.Mul(0),
		bex23.DevPtr(0), bex23.Mul(0),
		bex32.DevPtr(0), bex32.Mul(0),
		N, cfg)
}

func AddExchangeAFNCll(dst1, dst2, dst3, m1, m2, m3 *data.Slice, ms1, ms2, ms3 MSlice, llex12_red, llex13_red, llex23_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	//func AddExchange(B, m *data.Slice, Aex_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(1 / (c[X] * c[X]))
	wy := float32(1 / (c[Y] * c[Y]))
	wz := float32(1 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)

	//Sublattice 1
	k_addexchange_async(dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		ms1.DevPtr(0), ms1.Mul(0),
		unsafe.Pointer(llex12_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
	k_addexchange_async(dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		ms1.DevPtr(0), ms1.Mul(0),
		unsafe.Pointer(llex13_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)

	//Sublattice 2
	k_addexchange_async(dst2.DevPtr(X), dst2.DevPtr(Y), dst2.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		ms2.DevPtr(0), ms2.Mul(0),
		unsafe.Pointer(llex12_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
	k_addexchange_async(dst2.DevPtr(X), dst2.DevPtr(Y), dst2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		ms2.DevPtr(0), ms2.Mul(0),
		unsafe.Pointer(llex23_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)

	//Sublattice 3
	k_addexchange_async(dst3.DevPtr(X), dst3.DevPtr(Y), dst3.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		ms3.DevPtr(0), ms3.Mul(0),
		unsafe.Pointer(llex13_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
	k_addexchange_async(dst3.DevPtr(X), dst3.DevPtr(Y), dst3.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		ms3.DevPtr(0), ms3.Mul(0),
		unsafe.Pointer(llex23_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)

	/*
		k_addexchangeAfll_async(dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
			dst2.DevPtr(X), dst2.DevPtr(Y), dst2.DevPtr(Z),
			m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
			m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
			ms1.DevPtr(0), ms1.Mul(0),
			ms2.DevPtr(0), ms2.Mul(0),
			unsafe.Pointer(llex_red), regions.Ptr,
			wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)*/
}

func MagnetizationAFNC(dst, m1, m2, m3 *data.Slice, Ms1, Ms2, Ms3 MSlice) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_MagnetizationAFNC_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		Ms3.DevPtr(0), Ms3.Mul(0), N, cfg)
}

func NeelnAFNC(dst, m1, m2, m3 *data.Slice, Ms1, Ms2, Ms3 MSlice) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_NeelnAFNC_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		Ms3.DevPtr(0), Ms3.Mul(0), N, cfg)
}

func NeellAFNC(dst, m1, m2, m3 *data.Slice, Ms1, Ms2, Ms3 MSlice) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_NeellAFNC_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		Ms3.DevPtr(0), Ms3.Mul(0), N, cfg)
}

func GMagnetizationAFNC(dst, m1, m2, m3 *data.Slice, Ms1, Ms2, Ms3 MSlice, g1, g2, g3 float32) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_GMagnetizationAFNC_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		Ms3.DevPtr(0), Ms3.Mul(0),
		float32(g1), float32(g2), float32(g3),
		N, cfg)
}

func GNeelnAFNC(dst, m1, m2, m3 *data.Slice, Ms1, Ms2, Ms3 MSlice, g1, g2, g3 float32) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_GNeelnAFNC_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		Ms3.DevPtr(0), Ms3.Mul(0),
		float32(g1), float32(g2), float32(g3),
		N, cfg)
}

func GNeellAFNC(dst, m1, m2, m3 *data.Slice, Ms1, Ms2, Ms3 MSlice, g1, g2, g3 float32) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_GNeellAFNC_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		Ms3.DevPtr(0), Ms3.Mul(0),
		float32(g1), float32(g2), float32(g3),
		N, cfg)
}
