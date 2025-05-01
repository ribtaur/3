package engine

// Effective field

import "github.com/mumax/3/data"

var B_eff = NewVectorField("B_eff", "T", "Effective field", SetEffectiveField)

// Sets dst to the current effective field, in Tesla.
// This is the sum of all effective field terms,
// like demag, exchange, ...
func SetEffectiveField(dst *data.Slice) {
	SetDemagField(dst)    // set to B_demag...
	AddExchangeField(dst) // ...then add other terms
	AddAnisotropyField(dst)
	AddMagnetoelasticField(dst)
	AddAFMExchangeField(dst) // AFM Exchange non adjacent layers Victor mod
	B_ext.AddTo(dst)
	//print(OSC,"\n")
	/*if OSC == true {
		AddOSCField(dst)
	}*/
	if !relaxing {
		if LLBeq != true { // Needed not to add two times thermal noises in the case of LLB equation

			if LLBJHf == true {
				B_therm.LLBAddTo(dst)
			} else {
				B_therm.AddTo(dst)
			}
		}

	}
	AddCustomField(dst)
}
