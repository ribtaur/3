package engine

// Code for anitferro and no colinear antiferro

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

// Anisotropy, magnetization, exchange... variables and magnetization for sublattices
var (
	//Uniaxial anisotropy constants and vectors
	Ku11 = NewScalarParam("Ku11", "J/m3", "1st order uniaxial anisotropy constant lattice 1")
	Ku21 = NewScalarParam("Ku21", "J/m3", "2st order uniaxial anisotropy constant lattice 1")
	Ku12 = NewScalarParam("Ku12", "J/m3", "1st order uniaxial anisotropy constant lattice 2")
	Ku22 = NewScalarParam("Ku22", "J/m3", "2st order uniaxial anisotropy constant lattice 2")
	Ku13 = NewScalarParam("Ku13", "J/m3", "1st order uniaxial anisotropy constant lattice 3")
	Ku23 = NewScalarParam("Ku23", "J/m3", "2st order uniaxial anisotropy constant lattice 3")

	AnisU1           = NewVectorParam("anisU1", "", "Uniaxial anisotropy direction z lattice 1")
	AnisU2           = NewVectorParam("anisU2", "", "Uniaxial anisotropy direction z lattice 2")
	AnisU3           = NewVectorParam("anisU3", "", "Uniaxial anisotropy direction z lattice 3")
	AnisU1b           = NewVectorParam("anisU1b", "", "Uniaxial anisotropy direction x lattice 1")
	AnisU2b           = NewVectorParam("anisU2b", "", "Uniaxial anisotropy direction x lattice 2")
	AnisU3b           = NewVectorParam("anisU3b", "", "Uniaxial anisotropy direction x lattice 3")

	//Tetragonal anisotropy constants
	Ku21b = NewScalarParam("Ku21b", "J/m3", "2st order tetragonal anisotropy constant lattice 1")
	Ku22b = NewScalarParam("Ku22b", "J/m3", "2st order tetragonal anisotropy constant lattice 2")
	Ku23b = NewScalarParam("Ku23b", "J/m3", "2st order tetragonal anisotropy constant lattice 3")

	//Hexagonal anisotropy constants
	Ku31              = NewScalarParam("Ku31", "J/m3", "3rd order hexagonal anisotropy constant lattice 1")
	Ku31b             = NewScalarParam("Ku31b", "J/m3", "3rd order hexagonal anisotropy constant lattice 1")
	Ku32              = NewScalarParam("Ku32", "J/m3", "3rd order hexagonal anisotropy constant lattice 2")
	Ku32b             = NewScalarParam("Ku32b", "J/m3", "3rd order hexagonal anisotropy constant lattice 2")
	Ku33              = NewScalarParam("Ku33", "J/m3", "3rd order hexagonal anisotropy constant lattice 3")
	Ku33b             = NewScalarParam("Ku33b", "J/m3", "3rd order hexagonal anisotropy constant lattice 3")

/* Does not work like that anymore
	// Second set for another uniaxial Anisotropy
	Ku11b   = NewScalarParam("Ku11b", "J/m3", "1st order uniaxial anisotropy constant lattice 1")
	Ku21b   = NewScalarParam("Ku21b", "J/m3", "2st order uniaxial anisotropy constant lattice 1")
	Ku12b   = NewScalarParam("Ku12b", "J/m3", "1st order uniaxial anisotropy constant lattice 2")
	Ku22b   = NewScalarParam("Ku22b", "J/m3", "2st order uniaxial anisotropy constant lattice 2")
	AnisU1b = NewVectorParam("anisU1b", "", "Uniaxial anisotropy direction")
*/

	// Cubic anisotropy constants
	Kc11             = NewScalarParam("Kc11", "J/m3", "1st order cubic anisotropy constant lattice 1")
	Kc21             = NewScalarParam("Kc21", "J/m3", "2nd order cubic anisotropy constant lattice 1")
	Kc31             = NewScalarParam("Kc31", "J/m3", "3rd order cubic anisotropy constant lattice 1")
	Kc12             = NewScalarParam("Kc12", "J/m3", "1st order cubic anisotropy constant lattice 2")
	Kc22             = NewScalarParam("Kc22", "J/m3", "2nd order cubic anisotropy constant lattice 2")
	Kc32             = NewScalarParam("Kc32", "J/m3", "3rd order cubic anisotropy constant lattice 2")
	Kc13             = NewScalarParam("Kc13", "J/m3", "1st order cubic anisotropy constant lattice 3")
	Kc23             = NewScalarParam("Kc23", "J/m3", "2nd order cubic anisotropy constant lattice 3")
	Kc33             = NewScalarParam("Kc33", "J/m3", "3rd order cubic anisotropy constant lattice 3")

	AnisC11          = NewVectorParam("anisC11", "", "Cubic anisotropy direction #1 lattice 1")
	AnisC21          = NewVectorParam("anisC21", "", "Cubic anisotropy direction #2 lattice 1")

	AnisC12          = NewVectorParam("anisC12", "", "Cubic anisotropy direction #1 lattice 2")
	AnisC22          = NewVectorParam("anisC22", "", "Cubic anisotropy direction #2 lattice 2")

	AnisC13          = NewVectorParam("anisC13", "", "Cubic anisotropy direction #1 lattice 3")
	AnisC23          = NewVectorParam("anisC23", "", "Cubic anisotropy direction #2 lattice 3")

	M1               magnetization // reduced magnetization (unit length)
	M2               magnetization // reduced magnetization (unit length)
	M3               magnetization // reduced magnetization (unit length)
	Msat1                          = NewScalarParam("Msat1", "A/m", "Saturation magnetization")
	M_full1                        = NewVectorField("m_full1", "A/m", "Unnormalized magnetization", SetMFull1)
	Msat2                          = NewScalarParam("Msat2", "A/m", "Saturation magnetization")
	M_full2                        = NewVectorField("m_full2", "A/m", "Unnormalized magnetization", SetMFull2)
	Msat3                          = NewScalarParam("Msat3", "A/m", "Saturation magnetization")
	M_full3                        = NewVectorField("m_full3", "A/m", "Unnormalized magnetization", SetMFull3)
	GammaLL1         float64       = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	GammaLL2         float64       = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	GammaLL3         float64       = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Pol1                           = NewScalarParam("Pol1", "", "Electrical current polarization Zhang-Li Lattice 1")
	Pol2                           = NewScalarParam("Pol2", "", "Electrical current polarization Zhang-Li Lattice 2")
	Pol3                           = NewScalarParam("Pol3", "", "Electrical current polarization Zhang-Li Lattice 3")
	isolatedlattices bool          = false // Debug only
	Alpha1                         = NewScalarParam("alpha1", "", "Landau-Lifshitz damping constant, lattice 1")
	Alpha2                         = NewScalarParam("alpha2", "", "Landau-Lifshitz damping constant, lattice 2")
	Alpha3                         = NewScalarParam("alpha3", "", "Landau-Lifshitz damping constant, lattice 3")
	Xi1                            = NewScalarParam("xi1", "", "Non-adiabaticity of spin-transfer-torque STT lattice 1")
	Xi2                            = NewScalarParam("xi2", "", "Non-adiabaticity of spin-transfer-torque STT lattice 2")
	Xi3                            = NewScalarParam("xi3", "", "Non-adiabaticity of spin-transfer-torque STT lattice 3")
	PolSTT1                        = NewScalarParam("PolSTT1", "", "Electrical current polarization STT lattice 1")
	PolSTT2                        = NewScalarParam("PolSTT2", "", "Electrical current polarization STT lattice 2")
	PolSTT3                        = NewScalarParam("PolSTT3", "", "Electrical current polarization STT lattice 3")
	// For AF PRB 054401
	x_TM = NewScalarParam("x_TM", "a.u.", "TM ratio")
	nv   = NewScalarParam("nv", "a.u.", "Number of neighbours")
	mu1  = NewScalarParam("mu1", "J/T.", "Bohr magnetons lattice 1")
	mu2  = NewScalarParam("mu2", "J/T.", "Bohr magnetons lattice 2")
	J0aa = NewScalarParam("J0aa", "T.", "Exchange lattice 1")
	J0bb = NewScalarParam("J0bb", "T.", "Exchange lattice 2")
	J0ab = NewScalarParam("J0ab", "T.", "Exchange lattice 1-2")

	EpsilonPrime1 = NewScalarParam("EpsilonPrime1", "", "Slonczewski secondairy STT term ε' Lattice 1")
	EpsilonPrime2 = NewScalarParam("EpsilonPrime2", "", "Slonczewski secondairy STT term ε' Lattice 2")
	EpsilonPrime3 = NewScalarParam("EpsilonPrime3", "", "Slonczewski secondairy STT term ε' Lattice 3")

	// For LLB AF Angular momentum exchange (Unai)
	lambda0 = NewScalarParam("lambda0", "", "Moment exchange between sublattices")
	MFA     = false
	// For Brillouin
	Brillouin = false
	JA        = NewScalarParam("JA", "a.u.", "Billouin J lattice A")
	JB        = NewScalarParam("JB", "a.u.", "Billouin J lattice B")
	// Direct moment induction
	deltaM = NewScalarParam("deltaM", "a.u.", "Moment indiced by laser")

	// M and neel vectors, g stands for Gyromagnetic ratio
	MAFg  = NewVectorField("MAFg", "A/m", "Moment AFNC", GmagnetizationAFNC)
	n_Neelg = NewVectorField("Neelg_n", "A/m", "Neel vector n AFNC", GneelnAFNC)
	l_Neelg = NewVectorField("Neelg_l", "A/m", "Neel vector l AFNC", GneellAFNC)
	MAF   = NewVectorField("MAF", "A/m", "Magnetization AFNC", MagnetizationAFNC)
	n_Neel  = NewVectorField("Neel_n", "A/m", "Neel vector n AFNC", NeelnAFNC)
	l_Neel  = NewVectorField("Neel_l", "A/m", "Neel vector l AFNC", NeellAFNC)
)

func init() {
	DeclFunc("UpdateM", UpdateM, "UpdateM")
	DeclFunc("InitAntiferro", InitAntiferro, "InitAntiferro")
	DeclLValue("m1", &M1, `Reduced magnetization sublattice 1 (unit length)`)
	DeclLValue("m2", &M2, `Reduced magnetization sublattice 2 (unit length)`)
	DeclLValue("m3", &M3, `Reduced magnetization sublattice 3 (unit length)`)
	M1.name = "m1_"
	M2.name = "m2_"
	M3.name = "m3_"
	DeclVar("GammaLL1", &GammaLL1, "Gyromagnetic ratio in rad/Ts Lattice 1")
	DeclVar("GammaLL2", &GammaLL2, "Gyromagnetic ratio in rad/Ts Lattice 2")
	DeclVar("GammaLL3", &GammaLL3, "Gyromagnetic ratio in rad/Ts Lattice 3")
	DeclVar("isolatedlattices", &isolatedlattices, "Isolate AFNC lattices")
	DeclVar("MFA", &MFA, "MFA model for AF LLB")
	DeclVar("Brillouin", &Brillouin, "Brillouin model for AF LLB")
}

func InitAntiferro() {
	M1.alloc()
	M2.alloc()
	M3.alloc()
}

// full magnetization for each sublattice
func SetMFull1(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat1.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M1.Buffer().Comp(c), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := geometry.Slice()
	if rV {
		defer cuda.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

func SetMFull2(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat2.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M2.Buffer().Comp(c), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := geometry.Slice()
	if rV {
		defer cuda.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

func SetMFull3(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat3.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M3.Buffer().Comp(c), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := geometry.Slice()
	if rV {
		defer cuda.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

// FUNCTIONS FOR ANTIFERRO

func torqueFnAF(dst1, dst2 *data.Slice) {

	// Set Effective field

	if !isolatedlattices {
		UpdateM()
		SetDemagField(dst1)
		data.Copy(dst2, dst1)
	} else { // Just for code debug
		data.Copy(M.Buffer(), M1.Buffer())
		*Msat = *Msat1
		SetDemagField(dst1)
		data.Copy(M.Buffer(), M2.Buffer())
		*Msat = *Msat2
		SetDemagField(dst2)
	}
	AddExchangeFieldAF(dst1, dst2)
	AddAnisotropyFieldAF(dst1, dst2)
	//AddAFMExchangeField(dst)  // AFM Exchange non adjacent layers
	B_ext.AddTo(dst1)
	B_ext.AddTo(dst2)
	if !relaxing {
		if LLBeq != true {
			B_therm.AddToAF(dst1, dst2)
		}
	}
	AddCustomField(dst1)
	AddCustomField(dst2)

	// Add to sublattice 1 and 2
	alpha1 := Alpha1.MSlice()
	defer alpha1.Recycle()
	alpha2 := Alpha2.MSlice()
	defer alpha2.Recycle()
	if Precess {
		cuda.LLTorque(dst1, M1.Buffer(), dst1, alpha1) // overwrite dst with torque
		cuda.LLTorque(dst2, M2.Buffer(), dst2, alpha2)
	} else {
		cuda.LLNoPrecess(dst1, M1.Buffer(), dst1)
		cuda.LLNoPrecess(dst2, M2.Buffer(), dst2)
	}

	// STT
	AddSTTorqueAF(dst1, dst2)

	FreezeSpins(dst1)
	FreezeSpins(dst2)

	NEvals++
}

func AddSTTorqueAF(dst1, dst2 *data.Slice) {

	if J.isZero() {
		return
	}
	//util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	fl, rec := FixedLayer.Slice()
	if rec {
		defer cuda.Recycle(fl)
	}
	//Lattice 1 and 2
	if !DisableZhangLiTorque {
		msat1 := Msat1.MSlice()
		defer msat1.Recycle()
		msat2 := Msat2.MSlice()
		defer msat2.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		alpha1 := Alpha1.MSlice()
		defer alpha1.Recycle()
		alpha2 := Alpha2.MSlice()
		defer alpha2.Recycle()
		xi1 := Xi1.MSlice()
		defer xi1.Recycle()
		xi2 := Xi2.MSlice()
		defer xi2.Recycle()
		polstt1 := PolSTT1.MSlice()
		defer polstt1.Recycle()
		polstt2 := PolSTT2.MSlice()
		defer polstt2.Recycle()
		cuda.AddZhangLiTorque(dst1, M1.Buffer(), msat1, j, alpha1, xi1, polstt1, Mesh())
		cuda.AddZhangLiTorque(dst2, M2.Buffer(), msat2, j, alpha2, xi2, polstt2, Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {

		msat := Msat.MSlice()
		defer msat.Recycle()
		msat1 := Msat1.MSlice()
		defer msat1.Recycle()
		msat2 := Msat2.MSlice()
		defer msat2.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		fixedP := FixedLayer.MSlice()
		defer fixedP.Recycle()
		alpha1 := Alpha1.MSlice()
		defer alpha1.Recycle()
		alpha2 := Alpha2.MSlice()
		defer alpha2.Recycle()
		pol1 := Pol1.MSlice()
		defer pol1.Recycle()
		pol2 := Pol2.MSlice()
		defer pol2.Recycle()
		lambda := Lambda.MSlice()
		defer lambda.Recycle()
		//epsPrime := EpsilonPrime.MSlice()
		//defer epsPrime.Recycle()
		epsPrime1 := EpsilonPrime1.MSlice()
		defer epsPrime1.Recycle()
		epsPrime2 := EpsilonPrime2.MSlice()
		defer epsPrime2.Recycle()
		thickness := FreeLayerThickness.MSlice()
		defer thickness.Recycle()
		if LLBeq == false {
			cuda.AddSlonczewskiTorque2(dst1, M1.Buffer(),
				msat1, j, fixedP, alpha1, pol1, lambda, epsPrime1,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
			cuda.AddSlonczewskiTorque2(dst2, M2.Buffer(),
				msat2, j, fixedP, alpha2, pol2, lambda, epsPrime2,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
		} else {
			cuda.AddSlonczewskiTorque2LLB(dst1, M1.Buffer(),
				msat1, j, fixedP, alpha1, pol1, lambda, epsPrime1,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
			cuda.AddSlonczewskiTorque2LLB(dst2, M2.Buffer(),
				msat2, j, fixedP, alpha2, pol2, lambda, epsPrime2,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
		}

	}
}

// Thermal field
func (b *thermField) AddToAF(dst1, dst2 *data.Slice) {
	if !Temp.isZero() {
		b.updateAF(1)
		cuda.Add(dst1, dst1, b.noise)
		b.updateAF(2)
		cuda.Add(dst2, dst2, b.noise)
	}
}

func (b *thermField) updateAF(i int) {
	// we need to fix the time step here because solver will not yet have done it before the first step.
	// FixDt as an lvalue that sets Dt_si on change might be cleaner.

	if FixDt != 0 {
		Dt_si = FixDt
	}

	if b.generator == 0 {
		b.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		b.generator.SetSeed(b.seed)
	}
	if b.noise == nil {
		b.noise = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		// when noise was (re-)allocated it's invalid for sure.
		B_therm.step = -1
		B_therm.dt = -1
	}

	if Temp.isZero() {
		cuda.Memset(b.noise, 0, 0, 0)
		b.step = NSteps
		b.dt = Dt_si
		return
	}

	// keep constant during time step
	if NSteps == b.step && Dt_si == b.dt && solvertype < 6 {
		return
	}

	if FixDt == 0 {
		util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()
	k2_VgammaDt := 2 * mag.Kb / (GammaLL * cellVolume() * Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise

	ms := Msat1.MSlice()
	defer ms.Recycle()
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if i == 1 {
		ms = Msat1.MSlice()
		defer ms.Recycle()
		alpha = Alpha1.MSlice()
	}
	if i == 2 {
		ms = Msat2.MSlice()
		defer ms.Recycle()
		alpha = Alpha2.MSlice()
	}

	temp := Temp.MSlice()
	defer temp.Recycle()
	alpha0 := Alpha.MSlice()
	defer alpha0.Recycle()
	Noise_scale := 1.0
	if JHThermalnoise == false {
		Noise_scale = 0.0
	} else {
		Noise_scale = 1.0
	} // To cancel themal noise if needed
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt*Noise_scale, ms, temp, alpha0, ScaleNoiseLLB)
	}
	b.step = NSteps
	b.dt = Dt_si
}

func AddAnisotropyFieldAF(dst1, dst2 *data.Slice) {
	addUniaxialAnisotropyFrom(dst1, M1, Msat1, Ku11, Ku21, AnisU1)
	addUniaxialAnisotropyFrom(dst2, M2, Msat2, Ku12, Ku22, AnisU2)
	// second axis
	// addUniaxialAnisotropyFrom(dst1, M1, Msat1, Ku11b, Ku21b, AnisU1b)
	// addUniaxialAnisotropyFrom(dst2, M2, Msat2, Ku12b, Ku22b, AnisU1b)

	addCubicAnisotropyFrom(dst1, M1, Msat1, Kc11, Kc21, Kc31, AnisC11, AnisC21)
	addCubicAnisotropyFrom(dst2, M2, Msat2, Kc12, Kc22, Kc32, AnisC12, AnisC22)
}

func RenormAF(y01, y02 *data.Slice, dt, GammaLL1, GammaLL2 float32) {

	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	alpha1 := Alpha1.MSlice()
	defer alpha1.Recycle()
	alpha2 := Alpha2.MSlice()
	defer alpha2.Recycle()

	Tcurie := TCurie.MSlice()
	defer Tcurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()
	Msat1 := Msat1.MSlice()
	defer Msat1.Recycle()
	Msat2 := Msat2.MSlice()
	defer Msat2.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()

	X_TM := x_TM.MSlice()
	defer X_TM.Recycle()
	NV := nv.MSlice()
	defer NV.Recycle()

	MU1 := mu1.MSlice()
	defer MU1.Recycle()
	MU2 := mu2.MSlice()
	defer MU2.Recycle()

	J0AA := J0aa.MSlice()
	defer J0AA.Recycle()
	J0BB := J0bb.MSlice()
	defer J0BB.Recycle()
	J0AB := J0ab.MSlice()
	defer J0AB.Recycle()

	cuda.LLBRenormAF(y01, y02, M1.Buffer(), M2.Buffer(), temp, alpha, alpha1, alpha2, Tcurie, Msat, Msat1, Msat2, X_TM, NV, MU1, MU2, J0AA, J0BB, J0AB, dt, GammaLL1, GammaLL2)

}

func RenormAFBri(y01, y02 *data.Slice, dt, GammaLL1, GammaLL2 float32) {

	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	alpha1 := Alpha1.MSlice()
	defer alpha1.Recycle()
	alpha2 := Alpha2.MSlice()
	defer alpha2.Recycle()

	Tcurie := TCurie.MSlice()
	defer Tcurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()
	Msat1 := Msat1.MSlice()
	defer Msat1.Recycle()
	Msat2 := Msat2.MSlice()
	defer Msat2.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()

	X_TM := x_TM.MSlice()
	defer X_TM.Recycle()
	NV := nv.MSlice()
	defer NV.Recycle()

	MU1 := mu1.MSlice()
	defer MU1.Recycle()
	MU2 := mu2.MSlice()
	defer MU2.Recycle()

	J0AA := J0aa.MSlice()
	defer J0AA.Recycle()
	J0BB := J0bb.MSlice()
	defer J0BB.Recycle()
	J0AB := J0ab.MSlice()
	defer J0AB.Recycle()

	// for Brillouin
	Ja := JA.MSlice()
	defer Ja.Recycle()
	Jb := JB.MSlice()
	defer Jb.Recycle()

	cuda.LLBRenormAFBri(y01, y02, M1.Buffer(), M2.Buffer(), temp, alpha, alpha1, alpha2, Tcurie, Msat, Msat1, Msat2, X_TM, NV, MU1, MU2, J0AA, J0BB, J0AB, dt, GammaLL1, GammaLL2, Ja, Jb)

}

// FUNCTIONS FOR NO COLINEAR

// write torque to dst and increment NEvals
func torqueFnAFNC(dst1, dst2, dst3 *data.Slice) {

	// Set Effective field

	if !isolatedlattices {
		UpdateM()
		SetDemagField(dst1)
		data.Copy(dst2, dst1)
		data.Copy(dst3, dst1)
	} else { // Just for code debug
		data.Copy(M.Buffer(), M1.Buffer())
		*Msat = *Msat1
		SetDemagField(dst1)
		data.Copy(M.Buffer(), M2.Buffer())
		*Msat = *Msat2
		SetDemagField(dst2)
		data.Copy(M.Buffer(), M3.Buffer())
		*Msat = *Msat3
		SetDemagField(dst3)
	}
	AddExchangeFieldAFNC(dst1, dst2, dst3)
	AddAnisotropyFieldAFNC(dst1, dst2, dst3)
	//AddAFMExchangeField(dst)  // AFM Exchange non adjacent layers
	B_ext.AddTo(dst1)
	B_ext.AddTo(dst2)
	B_ext.AddTo(dst3)
	if !relaxing {
		if LLBeq != true {
			B_therm.AddToAFNC(dst1, dst2, dst3)
		}
	}
	AddCustomField(dst1)
	AddCustomField(dst2)
	AddCustomField(dst3)

	// Add to sublattice 1 and 2
	alpha1 := Alpha1.MSlice()
	defer alpha1.Recycle()
	alpha2 := Alpha2.MSlice()
	defer alpha2.Recycle()
	alpha3 := Alpha3.MSlice()
	defer alpha3.Recycle()
	if Precess {
		cuda.LLTorque(dst1, M1.Buffer(), dst1, alpha1) // overwrite dst with torque
		cuda.LLTorque(dst2, M2.Buffer(), dst2, alpha2)
		cuda.LLTorque(dst3, M3.Buffer(), dst3, alpha3)
	} else {
		cuda.LLNoPrecess(dst1, M1.Buffer(), dst1)
		cuda.LLNoPrecess(dst2, M2.Buffer(), dst2)
		cuda.LLNoPrecess(dst3, M3.Buffer(), dst3)
	}

	// STT
	AddSTTorqueAFNC(dst1, dst2, dst3)

	FreezeSpins(dst1)
	FreezeSpins(dst2)
	FreezeSpins(dst3)

	NEvals++
}

func AddSTTorqueAFNC(dst1, dst2, dst3 *data.Slice) {

	if J.isZero() {
		return
	}
	//util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	fl, rec := FixedLayer.Slice()
	if rec {
		defer cuda.Recycle(fl)
	}
	// Lattices 1, 2 and 3
	if !DisableZhangLiTorque {
		msat1 := Msat1.MSlice()
		defer msat1.Recycle()
		msat2 := Msat2.MSlice()
		defer msat2.Recycle()
		msat3 := Msat3.MSlice()
		defer msat3.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		alpha1 := Alpha1.MSlice()
		defer alpha1.Recycle()
		alpha2 := Alpha2.MSlice()
		defer alpha2.Recycle()
		alpha3 := Alpha3.MSlice()
		defer alpha3.Recycle()
		xi1 := Xi1.MSlice()
		defer xi1.Recycle()
		xi2 := Xi2.MSlice()
		defer xi2.Recycle()
		xi3 := Xi3.MSlice()
		defer xi3.Recycle()
		polstt1 := PolSTT1.MSlice()
		defer polstt1.Recycle()
		polstt2 := PolSTT2.MSlice()
		defer polstt2.Recycle()
		polstt3 := PolSTT3.MSlice()
		defer polstt3.Recycle()
		cuda.AddZhangLiTorque(dst1, M1.Buffer(), msat1, j, alpha1, xi1, polstt1, Mesh())
		cuda.AddZhangLiTorque(dst2, M2.Buffer(), msat2, j, alpha2, xi2, polstt2, Mesh())
		cuda.AddZhangLiTorque(dst3, M3.Buffer(), msat3, j, alpha3, xi3, polstt3, Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {

		msat := Msat.MSlice()
		defer msat.Recycle()
		msat1 := Msat1.MSlice()
		defer msat1.Recycle()
		msat2 := Msat2.MSlice()
		defer msat2.Recycle()
		msat3 := Msat3.MSlice()
		defer msat3.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		fixedP := FixedLayer.MSlice()
		defer fixedP.Recycle()
		alpha1 := Alpha1.MSlice()
		defer alpha1.Recycle()
		alpha2 := Alpha2.MSlice()
		defer alpha2.Recycle()
		alpha3 := Alpha3.MSlice()
		defer alpha3.Recycle()
		pol1 := Pol1.MSlice()
		defer pol1.Recycle()
		pol2 := Pol2.MSlice()
		defer pol2.Recycle()
		pol3 := Pol3.MSlice()
		defer pol3.Recycle()
		lambda := Lambda.MSlice()
		defer lambda.Recycle()
		//epsPrime := EpsilonPrime.MSlice()
		//defer epsPrime.Recycle()
		epsPrime1 := EpsilonPrime1.MSlice()
		defer epsPrime1.Recycle()
		epsPrime2 := EpsilonPrime2.MSlice()
		defer epsPrime2.Recycle()
		epsPrime3 := EpsilonPrime3.MSlice()
		defer epsPrime3.Recycle()
		thickness := FreeLayerThickness.MSlice()
		defer thickness.Recycle()
		if LLBeq == false {
			cuda.AddSlonczewskiTorque2(dst1, M1.Buffer(),
				msat1, j, fixedP, alpha1, pol1, lambda, epsPrime1,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
			cuda.AddSlonczewskiTorque2(dst2, M2.Buffer(),
				msat2, j, fixedP, alpha2, pol2, lambda, epsPrime2,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
			cuda.AddSlonczewskiTorque2(dst3, M3.Buffer(),
				msat3, j, fixedP, alpha3, pol3, lambda, epsPrime3,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
		} else {
			cuda.AddSlonczewskiTorque2LLB(dst1, M1.Buffer(),
				msat1, j, fixedP, alpha1, pol1, lambda, epsPrime1,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
			cuda.AddSlonczewskiTorque2LLB(dst2, M2.Buffer(),
				msat2, j, fixedP, alpha2, pol2, lambda, epsPrime2,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
			cuda.AddSlonczewskiTorque2LLB(dst3, M3.Buffer(),
				msat3, j, fixedP, alpha3, pol3, lambda, epsPrime3,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
		}

	}
}

// Thermal field
func (b *thermField) AddToAFNC(dst1, dst2, dst3 *data.Slice) {
	if !Temp.isZero() {
		b.updateAFNC(1)
		cuda.Add(dst1, dst1, b.noise)
		b.updateAFNC(2)
		cuda.Add(dst2, dst2, b.noise)
		b.updateAFNC(3)
		cuda.Add(dst3, dst3, b.noise)
	}
}

func (b *thermField) updateAFNC(i int) {
	// we need to fix the time step here because solver will not yet have done it before the first step.
	// FixDt as an lvalue that sets Dt_si on change might be cleaner.

	if FixDt != 0 {
		Dt_si = FixDt
	}

	if b.generator == 0 {
		b.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		b.generator.SetSeed(b.seed)
	}
	if b.noise == nil {
		b.noise = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		// when noise was (re-)allocated it's invalid for sure.
		B_therm.step = -1
		B_therm.dt = -1
	}

	if Temp.isZero() {
		cuda.Memset(b.noise, 0, 0, 0)
		b.step = NSteps
		b.dt = Dt_si
		return
	}

	// keep constant during time step
	if NSteps == b.step && Dt_si == b.dt && solvertype < 6 {
		return
	}

	if FixDt == 0 {
		util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()
	k2_VgammaDt := 2 * mag.Kb / (GammaLL * cellVolume() * Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise

	ms := Msat1.MSlice() // Redundant
	defer ms.Recycle()
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if i == 1 {
		ms = Msat1.MSlice()
		defer ms.Recycle()
		alpha = Alpha1.MSlice() // No deferring alpha?
	}
	if i == 2 {
		ms = Msat2.MSlice()
		defer ms.Recycle()
		alpha = Alpha2.MSlice()
	}
	if i == 3 {
		ms = Msat3.MSlice()
		defer ms.Recycle()
		alpha = Alpha3.MSlice()
	}

	temp := Temp.MSlice()
	defer temp.Recycle()
	alpha0 := Alpha.MSlice()
	defer alpha0.Recycle()
	Noise_scale := 1.0
	if JHThermalnoise == false {
		Noise_scale = 0.0
	} else {
		Noise_scale = 1.0
	} // To cancel themal noise if needed
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt*Noise_scale, ms, temp, alpha0, ScaleNoiseLLB)
	}
	b.step = NSteps
	b.dt = Dt_si
}

func AddAnisotropyFieldAFNC(dst1, dst2, dst3 *data.Slice) {

	// Sanity check for anisotropies, only first sublattice is considered.
	// And since tetragonal and hexagonal share terms with uniaxial, the
	// only conflicting cases are (Uniaxial,Cubic) and (Tetragonal,Hexagonal)

	haveUniaxial := Ku11.nonZero() || Ku21.nonZero()
	haveCubic := Kc11.nonZero() || Kc21.nonZero() || Kc31.nonZero()
	haveTetrag := Ku21b.nonZero()
	haveHexag := Ku31.nonZero() || Ku31b.nonZero()

	if haveUniaxial && haveCubic {
		util.Fatal("Cannot have both uniaxial and cubic anisotropy")
	}

	if haveTetrag && haveHexag {
		util.Fatal("Cannot have both tetragonal and hexagonal anisotropy")
	}

	addUniaxialAnisotropyFrom(dst1, M1, Msat1, Ku11, Ku21, AnisU1)
	addUniaxialAnisotropyFrom(dst2, M2, Msat2, Ku12, Ku22, AnisU2)
	addUniaxialAnisotropyFrom(dst3, M3, Msat3, Ku13, Ku23, AnisU3)

	addTetragonalAnisotropyFrom(dst1, M1, Msat1, Ku21b, AnisU1, AnisU1b)
	addTetragonalAnisotropyFrom(dst2, M2, Msat2, Ku22b, AnisU2, AnisU2b)
	addTetragonalAnisotropyFrom(dst3, M3, Msat3, Ku23b, AnisU3, AnisU3b)

	addHexagonalAnisotropyFrom(dst1, M1 , Msat1, Ku31, Ku31b, AnisU1, AnisU1b)
	addHexagonalAnisotropyFrom(dst2, M2 , Msat2, Ku32, Ku32b, AnisU2, AnisU2b)
	addHexagonalAnisotropyFrom(dst3, M3 , Msat3, Ku33, Ku33b, AnisU3, AnisU3b)

	addCubicAnisotropyFrom(dst1, M1, Msat1, Kc11, Kc21, Kc31, AnisC11, AnisC21)
	addCubicAnisotropyFrom(dst2, M2, Msat2, Kc12, Kc22, Kc32, AnisC12, AnisC22)
	addCubicAnisotropyFrom(dst3, M3, Msat3, Kc13, Kc23, Kc33, AnisC13, AnisC23)

}

func UpdateM() {
	ms0 := Msat.MSlice()
	defer ms0.Recycle()
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	ms3 := Msat3.MSlice()
	defer ms3.Recycle()
	cuda.NormalizeAFNC(M.Buffer(), M1.Buffer(), M2.Buffer(), M3.Buffer(), ms0, ms1, ms2, ms3)
}

func MagnetizationAFNC(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	ms3 := Msat3.MSlice()
	defer ms3.Recycle()
	cuda.MagnetizationAFNC(dst, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3)
}

func NeelnAFNC(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	ms3 := Msat3.MSlice()
	defer ms3.Recycle()
	cuda.NeelnAFNC(dst, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3)
}

func NeellAFNC(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	ms3 := Msat3.MSlice()
	defer ms3.Recycle()
	cuda.NeellAFNC(dst, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3)
}

func GmagnetizationAFNC(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	ms3 := Msat3.MSlice()
	defer ms3.Recycle()
	cuda.GMagnetizationAFNC(dst, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3, float32(GammaLL1), float32(GammaLL2), float32(GammaLL3))
}

func GneelnAFNC(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	ms3 := Msat3.MSlice()
	defer ms3.Recycle()
	cuda.GNeelnAFNC(dst, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3, float32(GammaLL1), float32(GammaLL2), float32(GammaLL3))
}

func GneellAFNC(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	ms3 := Msat3.MSlice()
	defer ms3.Recycle()
	cuda.GNeellAFNC(dst, M1.Buffer(), M2.Buffer(), M3.Buffer(), ms1, ms2, ms3, float32(GammaLL1), float32(GammaLL2), float32(GammaLL3))
}
