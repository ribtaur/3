package engine

// Exchange interaction (Heisenberg + Dzyaloshinskii-Moriya) for AFNC implementation
// See also cuda/exchange.cu, cuda/ext_AFNC_dmi.cu and cuda/ext_AFNC_AntiferroNC.go

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Bex12 = NewScalarParam("Bex12", "J/m", "Exchange stiffness interlattice same cell 1-2")
	Bex21 = NewScalarParam("Bex21", "J/m", "Exchange stiffness interlattice same cell 2-1")
	Bex13 = NewScalarParam("Bex13", "J/m", "Exchange stiffness interlattice same cell 1-3")
	Bex31 = NewScalarParam("Bex31", "J/m", "Exchange stiffness interlattice same cell 3-1")
	Bex23 = NewScalarParam("Bex23", "J/m", "Exchange stiffness interlattice same cell 2-3")
	Bex32 = NewScalarParam("Bex32", "J/m", "Exchange stiffness interlattice same cell 3-2")

	Aexll12 = NewScalarParam("Aexll12", "J/m", "Exchange stiffness interlattice different cell 1-2", &lexll12)
	Aexll13 = NewScalarParam("Aexll13", "J/m", "Exchange stiffness interlattice different cell 1-3", &lexll13)
	Aexll23 = NewScalarParam("Aexll23", "J/m", "Exchange stiffness interlattice different cell 2-3", &lexll23)

	Aex1   = NewScalarParam("Aex1", "J/m", "Exchange stiffness lattice 1", &lex21)
	Dind1  = NewScalarParam("Dind1", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength lattice 1", &din21)
	Dbulk1 = NewScalarParam("Dbulk1", "J/m2", "Bulk Dzyaloshinskii-Moriya strength lattice 1", &dbulk21)

	Aex2   = NewScalarParam("Aex2", "J/m", "Exchange stiffness lattice 2", &lex22)
	Dind2  = NewScalarParam("Dind2", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength lattice 2", &din22)
	Dbulk2 = NewScalarParam("Dbulk2", "J/m2", "Bulk Dzyaloshinskii-Moriya strength lattice 2", &dbulk22)

	Aex3   = NewScalarParam("Aex3", "J/m", "Exchange stiffness lattice 3", &lex23)
	Dind3  = NewScalarParam("Dind3", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength lattice 3", &din23)
	Dbulk3 = NewScalarParam("Dbulk3", "J/m2", "Bulk Dzyaloshinskii-Moriya strength lattice 3", &dbulk23)

	lexll12 exchParam // inter-cell exchange 1-2
	lexll13 exchParam // inter-cell exchange 1-3
	lexll23 exchParam // inter-cell exchange 2-3

	lex21   exchParam // inter-cell exchange
	din21   exchParam // inter-cell interfacial DMI
	dbulk21 exchParam // inter-cell bulk DMI

	lex22   exchParam // inter-cell exchange
	din22   exchParam // inter-cell interfacial DMI
	dbulk22 exchParam // inter-cell bulk DMI

	lex23   exchParam // inter-cell exchange
	din23   exchParam // inter-cell interfacial DMI
	dbulk23 exchParam // inter-cell bulk DMI

	// Trial Exchange Fields
	B_exch1 = NewVectorField("B_exch1", "T", "Exchange field Subnet 1", AddExchangeField1)
	B_exch2 = NewVectorField("B_exch2", "T", "Exchange field Subnet 2", AddExchangeField2)
	B_exch3 = NewVectorField("B_exch3", "T", "Exchange field Subnet 3", AddExchangeField3)
)

func init() {
	lex21.init(Aex1)
	din21.init(Dind1)
	dbulk21.init(Dbulk1)

	lex22.init(Aex2)
	din22.init(Dind2)
	dbulk22.init(Dbulk2)

	lex23.init(Aex3)
	din23.init(Dind3)
	dbulk23.init(Dbulk3)

	lexll12.init(Aexll12)
	lexll13.init(Aexll13)
	lexll23.init(Aexll23)

	DeclFunc("ext_InterExchangeAex1", InterExchangeAex1, "Sets exchange coupling between two regions, sublattice 1.")
	DeclFunc("ext_InterExchangeAex2", InterExchangeAex2, "Sets exchange coupling between two regions, sublattice 2.")
	DeclFunc("ext_InterExchangeAex3", InterExchangeAex2, "Sets exchange coupling between two regions, sublattice 3.")
	DeclFunc("ext_InterExchangeAexll12", InterExchangeAexll12, "Sets exchange coupling between two regions, sublattices 1-2.")
	DeclFunc("ext_InterExchangeAexll13", InterExchangeAexll13, "Sets exchange coupling between two regions, sublattices 1-3.")
	DeclFunc("ext_InterExchangeAexll23", InterExchangeAexll23, "Sets exchange coupling between two regions, sublattices 2-3.")
	DeclFunc("ext_ScaleExchangeAex1", ScaleInterExchangeAex1, "Re-scales exchange coupling between two regions, sublattice 1.")
	DeclFunc("ext_ScaleExchangeAex2", ScaleInterExchangeAex2, "Re-scales exchange coupling between two regions, sublattice 2.")
	DeclFunc("ext_ScaleExchangeAex3", ScaleInterExchangeAex3, "Re-scales exchange coupling between two regions, sublattice 3.")
	DeclFunc("ext_ScaleExchangeAexll12", ScaleInterExchangeAexll12, "Re-scales exchange coupling between two regions, sublattices 1-2.")
	DeclFunc("ext_ScaleExchangeAexll13", ScaleInterExchangeAexll13, "Re-scales exchange coupling between two regions, sublattices 1-3.")
	DeclFunc("ext_ScaleExchangeAexll23", ScaleInterExchangeAexll23, "Re-scales exchange coupling between two regions, sublattices 2-3.")
}

func AddExchangeFieldAF(dst1, dst2 *data.Slice) {

	// Write eveything new to use both subnets
	//Sublattice 1
	inter1 := !Dind1.isZero()
	bulk1 := !Dbulk1.isZero()
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	//Sublattice 2
	//inter2 := !Dind1.isZero()
	//bulk2 := !Dbulk1.isZero()
	ms2 := Msat1.MSlice()
	defer ms1.Recycle()
	bex12 := Bex12.MSlice()
	defer bex12.Recycle()
	bex21 := Bex21.MSlice()
	defer bex21.Recycle()

	// Asumo que manda la red 1 si se ponen parámetros cuzados bulk dmi se va todo al carajo by now

	switch {
	case !inter1 && !bulk1:
		cuda.AddExchange(dst1, M1.Buffer(), lex21.Gpu(), ms1, regions.Gpu(), M.Mesh())
		cuda.AddExchange(dst2, M2.Buffer(), lex22.Gpu(), ms2, regions.Gpu(), M.Mesh())
		cuda.AddExchangeAFCell(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, bex12, bex21)
		cuda.AddExchangeAFll(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, lexll12.Gpu(), regions.Gpu(), M.Mesh())
	case inter1 && !bulk1: // Ahora es un 2x1 con todos los exchanges
		cuda.AddDMIAF(dst1, dst2, M1.Buffer(), M2.Buffer(), lex21.Gpu(), din21.Gpu(), lex22.Gpu(), din22.Gpu(), lexll12.Gpu(), ms1, ms2, bex12, bex21, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
	case bulk1 && !inter1:
		cuda.AddDMIBulk(dst1, M1.Buffer(), lex21.Gpu(), dbulk21.Gpu(), ms1, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		cuda.AddDMIBulk(dst2, M2.Buffer(), lex22.Gpu(), dbulk22.Gpu(), ms2, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		cuda.AddExchangeAFCell(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, bex12, bex21)
		cuda.AddExchangeAFll(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, lexll12.Gpu(), regions.Gpu(), M.Mesh())
	case inter1 && bulk1:
		util.Fatal("Cannot have induced and interfacial DMI at the same time")

	}

}

// Adds the current exchange AFfield to dst
func AddExchangeFieldAFNC(dst1, dst2, dst3 *data.Slice) {

	//Sublattice 1
	inter1 := !Dind1.isZero()
	bulk1 := !Dbulk1.isZero()
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	//Sublattice 2
	//inter2 := !Dind2.isZero()
	//bulk2 := !Dbulk2.isZero()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()

	//Sublattice 3
	//inter3 := !Dind3.isZero()
	//bulk3 := !Dbulk3.isZero()
	ms3 := Msat3.MSlice()
	defer ms3.Recycle()

	bex12 := Bex12.MSlice()
	defer bex12.Recycle()
	bex21 := Bex21.MSlice()
	defer bex21.Recycle()
	bex13 := Bex13.MSlice()
	defer bex13.Recycle()
	bex31 := Bex31.MSlice()
	defer bex31.Recycle()
	bex23 := Bex23.MSlice()
	defer bex23.Recycle()
	bex32 := Bex32.MSlice()
	defer bex32.Recycle()

	// Asumo que manda la red 1 si se ponen parámetros cuzados bulk dmi se va todo al carajo by now

	switch {
	case !inter1 && !bulk1:
		cuda.AddExchange(dst1, M1.Buffer(), lex21.Gpu(), ms1, regions.Gpu(), M.Mesh())
		cuda.AddExchange(dst2, M2.Buffer(), lex22.Gpu(), ms2, regions.Gpu(), M.Mesh())
		cuda.AddExchange(dst3, M3.Buffer(), lex23.Gpu(), ms3, regions.Gpu(), M.Mesh())
		cuda.AddExchangeAFNCCell(dst1, dst2, dst3, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3, bex12, bex21, bex13, bex31, bex23, bex32)
		cuda.AddExchangeAFNCll(dst1, dst2, dst3, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3, lexll12.Gpu(), lexll13.Gpu(), lexll23.Gpu(), regions.Gpu(), M.Mesh())
	case inter1 && !bulk1: // DMIAFNC already considers intercell exchange
		cuda.AddDMIAFNC(dst1, dst2, dst3, M1.Buffer(), M2.Buffer(), M3.Buffer(), lex21.Gpu(), din21.Gpu(), lex22.Gpu(), din22.Gpu(), lex23.Gpu(), din23.Gpu(), lexll12.Gpu(), lexll13.Gpu(), lexll23.Gpu(), ms1, ms2, ms3, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		cuda.AddExchangeAFNCCell(dst1, dst2, dst3, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3, bex12, bex21, bex13, bex31, bex23, bex32)
	case bulk1 && !inter1:
		cuda.AddDMIBulk(dst1, M1.Buffer(), lex21.Gpu(), dbulk21.Gpu(), ms1, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		cuda.AddDMIBulk(dst2, M2.Buffer(), lex22.Gpu(), dbulk22.Gpu(), ms2, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		cuda.AddDMIBulk(dst3, M3.Buffer(), lex23.Gpu(), dbulk23.Gpu(), ms3, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		cuda.AddExchangeAFNCCell(dst1, dst2, dst3, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3, bex12, bex21, bex13, bex31, bex23, bex32)
		cuda.AddExchangeAFNCll(dst1, dst2, dst3, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3, lexll12.Gpu(), lexll13.Gpu(), lexll23.Gpu(), regions.Gpu(), M.Mesh())
	case inter1 && bulk1:
		util.Fatal("Cannot have induced and interfacial DMI at the same time")

	}

}

// Scales the heisenberg exchange interaction between region1 and 2.l
// Scale = 1 means the harmonic mean over the regions of Aex.
func ScaleInterExchangeAex1(region1, region2 int, scale float64) {
	lex21.setScale(region1, region2, scale)
}

// Sets the exchange interaction between region 1 and 2.
func InterExchangeAex1(region1, region2 int, value float64) {
	lex21.setInter(region1, region2, value)
}

// Scales the heisenberg exchange interaction between region1 and 2.
// Scale = 1 means the harmonic mean over the regions of Aex.
func ScaleInterExchangeAex2(region1, region2 int, scale float64) {
	lex22.setScale(region1, region2, scale)
}

// Sets the exchange interaction between region 1 and 2.
func InterExchangeAex2(region1, region2 int, value float64) {
	lex22.setInter(region1, region2, value)
}

// Scales the heisenberg exchange interaction between region1 and 2.
// Scale = 1 means the harmonic mean over the regions of Aex.
func ScaleInterExchangeAex3(region1, region2 int, scale float64) {
	lex23.setScale(region1, region2, scale)
}

// Sets the exchange interaction between region 1 and 2.
func InterExchangeAex3(region1, region2 int, value float64) {
	lex23.setInter(region1, region2, value)
}

// Scales the heisenberg exchange interaction between region1 and 2.
// Scale = 1 means the harmonic mean over the regions of Aex.
func ScaleInterExchangeAexll12(region1, region2 int, scale float64) {
	lexll12.setScale(region1, region2, scale)
}

func ScaleInterExchangeAexll13(region1, region2 int, scale float64) {
	lexll13.setScale(region1, region2, scale)
}

func ScaleInterExchangeAexll23(region1, region2 int, scale float64) {
	lexll23.setScale(region1, region2, scale)
}

// Sets the exchange interaction between region 1 and 2, different cells, different sublattices
func InterExchangeAexll12(region1, region2 int, value float64) {
	lexll12.setInter(region1, region2, value)
}

func InterExchangeAexll13(region1, region2 int, value float64) {
	lexll13.setInter(region1, region2, value)
}

func InterExchangeAexll23(region1, region2 int, value float64) {
	lexll23.setInter(region1, region2, value)
}

// Adds the current exchange field to dst
func AddExchangeField1(dst *data.Slice) {
	size := dst.Size()
	dst2 := cuda.Buffer(3, size)
	defer cuda.Recycle(dst2)
//	AddExchangeFieldAF(dst, dst2)
}

// Adds the current exchange field to dst
func AddExchangeField2(dst *data.Slice) {
	size := dst.Size()
	dst2 := cuda.Buffer(3, size)
	defer cuda.Recycle(dst2)
//  AddExchangeFieldAF(dst2, dst)
}

// Adds the current exchange field to dst
func AddExchangeField3(dst *data.Slice) {
	size := dst.Size()
	dst2 := cuda.Buffer(3, size)
	defer cuda.Recycle(dst2)
//	AddExchangeFieldAF(dst2, dst)
}
