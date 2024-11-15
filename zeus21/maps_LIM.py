"""

Make LIM maps! For fun and science

Author: Sarah Libanore
BGU - November 2024

"""

from . import cosmology
from . import constants

import numpy as np
import powerbox as pbox
from scipy.interpolate import interp1d
from pyfftw import empty_aligned as empty

class CoevalMaps_LIM:
    "Class that calculates and keeps coeval maps, one z at a time."

    def __init__(self, LIM_coefficients, Power_Spectrum_LIM, z, Lbox=600, Nbox=200, KIND=None, seed=1605):
        'the KIND flag determines the kind of map you make. Options are:'
        'KIND = 0, only LIM lognormal. OK approximation'

        zlist = LIM_coefficients.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.rhoL_avg = LIM_coefficients.rhoL_avg[_iz]
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed
        self.z = zlist[_iz] #will be slightly different from z input

        klist = Power_Spectrum_LIM.klist_PS
        k3over2pi2 = klist**3/(2*np.pi**2)

        if (KIND == 0): #just LIM, ~gaussian
                
            PLIM = Power_Spectrum_LIM.Deltasq_LIM_lin[_iz]/k3over2pi2

            PLIMnorminterp = interp1d(klist,PLIM/self.rhoL_avg**2,fill_value=0.0,bounds_error=False)

            pb = pbox.PowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: PLIMnorminterp(k), 
                boxlength = self.Lbox,           
                seed = self.seed                
            )

            self.LIMmap = self.rhoL_avg * (1 + pb.delta_x() )
            self.deltamap = None

