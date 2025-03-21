"""

Make LIM maps! For fun and science

Author: Sarah Libanore
BGU - November 2024

"""

from . import cosmology
from . import constants
from . import maps as zeus21_maps

import numpy as np
import powerbox as pbox
from scipy.interpolate import interp1d
from pyfftw import empty_aligned as empty


# This function produces density and LIM maps analogously to the way Zeus21 produces the 21cm ones, using the excess power between linear and non linear power spectra
class CoevalMaps_LIM_zeuslike:
    "Class that calculates and keeps coeval maps, one z at a time."

    def __init__(self, LIM_coefficients, Power_Spectrum_LIM, Power_Spectrum, z, Lbox=600, Nbox=200, KIND=None, seed=1605):
        'the KIND flag determines the kind of map you make. Options are:'
        'KIND = 0, only LIM lognormal. OK approximation'

        zlist = LIM_coefficients.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.Inu_global = LIM_coefficients.Inu_avg[_iz]
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed
        self.z = zlist[_iz] #will be slightly different from z input

        klist = Power_Spectrum_LIM.klist_PS
        k3over2pi2 = klist**3/(2*np.pi**2)

        if (KIND == 0): #just LIM, ~gaussian
                
            PLIM = Power_Spectrum_LIM.Deltasq_LIM[_iz]/k3over2pi2

            PLIMnorminterp = interp1d(klist,PLIM/self.Inu_global**2,fill_value=0.0,bounds_error=False)

            pb = pbox.PowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: PLIMnorminterp(k), 
                boxlength = self.Lbox,           
                seed = self.seed                
            )

            self.LIMmap = self.Inu_global * (1 + pb.delta_x() )
            self.deltamap = None

        elif (KIND == 1):

            Pd = Power_Spectrum.Deltasq_d_lin[_iz,:]/k3over2pi2
            Pdinterp = interp1d(klist,Pd,fill_value=0.0,bounds_error=False)

            pb = pbox.PowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: Pdinterp(k), 
                boxlength = self.Lbox,           
                seed = self.seed               
            )

            self.deltamap = pb.delta_x() #density map, basis of this KIND of approach

            #then we make a map of the linear LIM fluctuation
            PdLIM = Power_Spectrum_LIM.Deltasq_dLIM[_iz]/k3over2pi2

            powerratioint = interp1d(klist,PdLIM/Pd,fill_value=0.0,bounds_error=False)

            deltak = pb.delta_k()

            powerratio = powerratioint(pb.k())
            LIMlin_k = powerratio * deltak
            self.LIMmaplin= self.Inu_global + zeus21_maps.powerboxCtoR(pb,mapkin = LIMlin_k)

            # !!! NON LINEAR CORRECTION !!!
            excesspowerLIM = (Power_Spectrum_LIM.Deltasq_LIM[_iz,:]-Power_Spectrum_LIM.Deltasq_LIM_lin[_iz,:])/k3over2pi2

            lognormpower = interp1d(klist,excesspowerLIM/self.Inu_global**2,fill_value=0.0,bounds_error=False)

            pbe = pbox.LogNormalPowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: lognormpower(k), 
                boxlength = self.Lbox,           
                seed = self.seed+1  # uncorrelated
            )

            self.LIMmapNL = self.Inu_global*pbe.delta_x()

            #and finally, just add them together!
            self.LIMmap_lin = self.LIMmaplin 
            self.LIMmap = self.LIMmaplin +  self.LIMmapNL

