package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Solver for Antiferro with RK4

type AntiferroNCRK4 struct {
}

func (_ *AntiferroNCRK4) Step() {

	// Reduplicamos a dos redes de momento independientes

	//m := M.Buffer()

	m1 := M1.Buffer()
	size := m1.Size()
	m2 := M2.Buffer()
	m3 := M3.Buffer()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	t0 := Time
	// backup magnetization
	m10 := cuda.Buffer(3, size)
	m20 := cuda.Buffer(3, size)
	m30 := cuda.Buffer(3, size)
	defer cuda.Recycle(m10)
	defer cuda.Recycle(m20)
	defer cuda.Recycle(m30)
	data.Copy(m10, m1)
	data.Copy(m20, m2)
	data.Copy(m30, m3)

	k11, k12, k13, k14 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	k21, k22, k23, k24 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	k31, k32, k33, k34 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(k11)
	defer cuda.Recycle(k12)
	defer cuda.Recycle(k13)
	defer cuda.Recycle(k14)
	defer cuda.Recycle(k21)
	defer cuda.Recycle(k22)
	defer cuda.Recycle(k23)
	defer cuda.Recycle(k24)
	defer cuda.Recycle(k31)
	defer cuda.Recycle(k32)
	defer cuda.Recycle(k33)
	defer cuda.Recycle(k34)

	h := float32(Dt_si * GammaLL)   // internal time step = Dt * gammaLL
	h1 := float32(Dt_si * GammaLL1) // internal time step = Dt * gammaLL
	h2 := float32(Dt_si * GammaLL2) // internal time step = Dt * gammaLL
	h3 := float32(Dt_si * GammaLL3) // internal time step = Dt * gammaLL

	// stage 1

	//data.Copy(m, m1)  // later to the effective magnetization to rewrite

	torqueFnAFNC(k11, k21, k31)

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m1, m1, k11, 1, (1./2.)*h1) // m = m*1 + k1*h/2
	cuda.Madd2(m2, m2, k21, 1, (1./2.)*h2) // m = m*1 + k1*h/2
	cuda.Madd2(m3, m3, k31, 1, (1./2.)*h3) // m = m*1 + k1*h/2
	M1.normalize()
	M2.normalize()
	M3.normalize()
	if RenormLLB == true { // Not implemented for AFNC
		RenormAFBri(m1, m2, float32(Dt_si), float32(GammaLL1), float32(GammaLL2))
	}
	torqueFnAFNC(k12, k22, k32)

	// stage 3
	cuda.Madd2(m1, m10, k12, 1, (1./2.)*h1) // m = m0*1 + k2*1/2
	cuda.Madd2(m2, m20, k22, 1, (1./2.)*h2) // m = m0*1 + k2*1/2
	cuda.Madd2(m3, m30, k32, 1, (1./2.)*h3) // m = m0*1 + k2*1/2
	M1.normalize()
	M2.normalize()
	M3.normalize()
	if RenormLLB == true { // Not implemented for AFNC
		RenormAFBri(m1, m2, float32(Dt_si), float32(GammaLL1), float32(GammaLL2))
	}
	torqueFnAFNC(k13, k23, k33)

	// stage 4
	Time = t0 + Dt_si
	cuda.Madd2(m1, m10, k13, 1, 1.*h1) // m = m0*1 + k3*1
	cuda.Madd2(m2, m20, k23, 1, 1.*h2) // m = m0*1 + k3*1
	cuda.Madd2(m3, m30, k33, 1, 1.*h3) // m = m0*1 + k3*1
	M1.normalize()
	M2.normalize()
	M3.normalize()
	if RenormLLB == true { // Not implemented for AFNC
		RenormAFBri(m1, m2, float32(Dt_si), float32(GammaLL1), float32(GammaLL2))
	}
	torqueFnAFNC(k14, k24, k34)

	err1 := cuda.MaxVecDiff(k11, k14) * float64(h)
	err2 := cuda.MaxVecDiff(k21, k24) * float64(h)
	err3 := cuda.MaxVecDiff(k31, k34) * float64(h)

	err := -1.0
	if err3 > err1 {
		if err3 > err2 {
			err = err3
		} else {
			err = err2
		}
	} else {
		if err2 > err1 {
			err = err2
		} else {
			err = err1
		}
	}

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution
		cuda.Madd5(m1, m10, k11, k12, k13, k14, 1, (1./6.)*h1, (1./3.)*h1, (1./3.)*h1, (1./6.)*h1)
		M1.normalize()
		cuda.Madd5(m2, m20, k21, k22, k23, k24, 1, (1./6.)*h2, (1./3.)*h2, (1./3.)*h2, (1./6.)*h2)
		M2.normalize()
		cuda.Madd5(m3, m30, k31, k32, k33, k23, 1, (1./6.)*h3, (1./3.)*h3, (1./3.)*h3, (1./6.)*h3)
		M3.normalize()
		if RenormLLB == true { // Not implemented for AFNC
			RenormAFBri(m1, m2, float32(Dt_si), float32(GammaLL1), float32(GammaLL2))
		}
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
		setLastErr(err)
		setMaxTorque(k14) // Why not k24 or k34? min of the three
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m1, m10)
		data.Copy(m2, m20)
		data.Copy(m3, m30)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./5.))
	}
}

func (_ *AntiferroNCRK4) Free() {}
