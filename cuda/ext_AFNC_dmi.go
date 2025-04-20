package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Add effective field of Dzyaloshinskii-Moriya interaction to Beff (Tesla).
// According to Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// See dmi.cu
func AddDMIAFNC(Beff1, Beff2, Beff3 *data.Slice, m1, m2, m3 *data.Slice, Aex1_red, Dex1_red, Aex2_red, Dex2_red, Aex3_red, Dex3_red, Aexll12_red, Aexll13_red, Aexll23_red SymmLUT, Msat1, Msat2, Msat3 MSlice, regions *Bytes, mesh *data.Mesh, OpenBC bool) {
	cellsize := mesh.CellSize()
	N := Beff1.Size()
	util.Argument(m1.Size() == N)
	cfg := make3DConf(N)
	var openBC byte
	if OpenBC {
		openBC = 1
	}

	k_adddmiAFNC_async(Beff1.DevPtr(X), Beff1.DevPtr(Y), Beff1.DevPtr(Z),
		Beff2.DevPtr(X), Beff2.DevPtr(Y), Beff2.DevPtr(Z),
		Beff3.DevPtr(X), Beff3.DevPtr(Y), Beff3.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		m3.DevPtr(X), m3.DevPtr(Y), m3.DevPtr(Z),
		Msat1.DevPtr(0), Msat1.Mul(0),
		Msat2.DevPtr(0), Msat2.Mul(0),
		Msat3.DevPtr(0), Msat3.Mul(0),
		unsafe.Pointer(Aex1_red), unsafe.Pointer(Dex1_red),
		unsafe.Pointer(Aex2_red), unsafe.Pointer(Dex2_red),
		unsafe.Pointer(Aex3_red), unsafe.Pointer(Dex3_red),
		unsafe.Pointer(Aexll12_red), unsafe.Pointer(Aexll13_red), unsafe.Pointer(Aexll23_red),
		regions.Ptr,
		float32(cellsize[X]), float32(cellsize[Y]), float32(cellsize[Z]), N[X], N[Y], N[Z], mesh.PBC_code(), openBC, cfg)
}
