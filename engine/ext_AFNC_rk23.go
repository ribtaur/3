package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Bogacki-Shampine solver. 3rd order, 3 evaluations per step, adaptive step.
// 	http://en.wikipedia.org/wiki/Bogacki-Shampine_method
//
// 	k1 = f(tn, yn)
// 	k2 = f(tn + 1/2 h, yn + 1/2 h k1)
// 	k3 = f(tn + 3/4 h, yn + 3/4 h k2)
// 	y{n+1}  = yn + 2/9 h k1 + 1/3 h k2 + 4/9 h k3            // 3rd order
// 	k4 = f(tn + h, y{n+1})
// 	z{n+1} = yn + 7/24 h k1 + 1/4 h k2 + 1/3 h k3 + 1/8 h k4 // 2nd order
type AntiferroNCRK23 struct {
	k11 *data.Slice // torque at end of step is kept for beginning of next step
	k21 *data.Slice // torque at end of step is kept for beginning of next step
	k31 *data.Slice // torque at end of step is kept for beginning of next step
}

func (rk *AntiferroNCRK23) Step() {
	m1 := M1.Buffer()
	size := m1.Size()
	m2 := M2.Buffer()
	m3 := M3.Buffer()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// upon resize: remove wrongly sized k1
	if rk.k11.Size() != m1.Size() { // Not for k21 and k31?
		rk.Free()
	}

	// first step ever: one-time k1 init and eval
	if rk.k11 == nil {
		rk.k11 = cuda.NewSlice(3, size)
		rk.k21 = cuda.NewSlice(3, size)
		rk.k31 = cuda.NewSlice(3, size)
		torqueFnAFNC(rk.k11, rk.k21, rk.k31)
	}

	// FSAL cannot be used with temperature
	if !Temp.isZero() {
		torqueFnAFNC(rk.k11, rk.k21, rk.k31)
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

	k12, k13, k14 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	k22, k23, k24 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	k32, k33, k34 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k12)
	defer cuda.Recycle(k13)
	defer cuda.Recycle(k14)
	defer cuda.Recycle(k22)
	defer cuda.Recycle(k23)
	defer cuda.Recycle(k24)
	defer cuda.Recycle(k32)
	defer cuda.Recycle(k33)
	defer cuda.Recycle(k34)

	h := float32(Dt_si * GammaLL)   // internal time step = Dt * gammaLL
	h1 := float32(Dt_si * GammaLL1) // internal time step = Dt * gammaLL
	h2 := float32(Dt_si * GammaLL2) // internal time step = Dt * gammaLL
	h3 := float32(Dt_si * GammaLL3) // internal time step = Dt * gammaLL

	// there is no explicit stage 1: k1 from previous step

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m1, m1, rk.k11, 1, (1./2.)*h1) // m = m*1 + k1*h/2
	M1.normalize()
	cuda.Madd2(m2, m2, rk.k21, 1, (1./2.)*h2) // m = m*1 + k1*h/2
	M2.normalize()
	cuda.Madd2(m3, m3, rk.k31, 1, (1./2.)*h3) // m = m*1 + k1*h/2
	M3.normalize()
	torqueFnAFNC(k12, k22, k32)

	// stage 3
	Time = t0 + (3./4.)*Dt_si
	cuda.Madd2(m1, m10, k12, 1, (3./4.)*h1) // m = m0*1 + k2*3/4
	M1.normalize()
	cuda.Madd2(m2, m20, k22, 1, (3./4.)*h2) // m = m0*1 + k2*3/4
	M2.normalize()
	cuda.Madd2(m3, m30, k32, 1, (3./4.)*h3) // m = m0*1 + k2*3/4
	M3.normalize()
	torqueFnAFNC(k13, k23, k33)

	// 3rd order solution
	cuda.Madd4(m1, m10, rk.k11, k12, k13, 1, (2./9.)*h1, (1./3.)*h1, (4./9.)*h1)
	M1.normalize()
	cuda.Madd4(m2, m20, rk.k21, k22, k23, 1, (2./9.)*h2, (1./3.)*h2, (4./9.)*h2)
	M2.normalize()
	cuda.Madd4(m3, m30, rk.k31, k32, k33, 1, (2./9.)*h3, (1./3.)*h3, (4./9.)*h3)
	M3.normalize()

	// error estimate
	Time = t0 + Dt_si
	torqueFnAFNC(k14, k24, k34)
	Err1 := k12 // re-use k2 as error
	Err2 := k22 // re-use k2 as error
	Err3 := k32 // re-use k2 as error
	// difference of 3rd and 2nd order torque without explicitly storing them first
	cuda.Madd4(Err1, rk.k11, k12, k13, k14, (7./24.)-(2./9.), (1./4.)-(1./3.), (1./3.)-(4./9.), (1. / 8.))
	cuda.Madd4(Err2, rk.k21, k22, k23, k24, (7./24.)-(2./9.), (1./4.)-(1./3.), (1./3.)-(4./9.), (1. / 8.))
	cuda.Madd4(Err3, rk.k31, k32, k33, k34, (7./24.)-(2./9.), (1./4.)-(1./3.), (1./3.)-(4./9.), (1. / 8.))

	// determine error
	err1 := cuda.MaxVecNorm(Err1) * float64(h)
	err2 := cuda.MaxVecNorm(Err2) * float64(h)
	err3 := cuda.MaxVecNorm(Err3) * float64(h)
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
		setLastErr(err)
		setMaxTorque(k14)
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./3.))
		data.Copy(rk.k11, k14) // FSAL
		data.Copy(rk.k21, k24) // FSAL
		data.Copy(rk.k31, k34) // FSAL
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m1, m10)
		data.Copy(m2, m20)
		data.Copy(m3, m30)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
	}
}

func (rk *AntiferroNCRK23) Free() {
	rk.k11.Free()
	rk.k11 = nil
	rk.k21.Free()
	rk.k21 = nil
	rk.k31.Free()
	rk.k31 = nil
}
