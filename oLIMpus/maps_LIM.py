"""

Make LIM maps! For fun and science

Author: Sarah Libanore
BGU - November 2024

"""

import time 

from zeus21 import maps as zeus21_maps
from zeus21 import sfrd
from zeus21 import BMF, z21_utilities
from .LIM import LineLuminosity, get_LIM_coefficients
from zeus21.correlations import Correlations, Power_Spectra
from .correlations_LIM import Correlations_LIM, Power_Spectra_LIM

import numpy as np
import powerbox as pbox
from scipy.interpolate import interp1d
from pyfftw import empty_aligned as empty
import os
import pickle

import pyfftw
pyfftw.config.NUM_THREADS = 6

import numexpr as ne

    
# This function produces density and LIM maps analogously to the way Zeus21 produces the 21cm ones, using the excess power between linear and non linear power spectra
class CoevalBox_LIM_zeuslike:
    "Class that calculates and keeps coeval maps, one z at a time."
    "The computation is done analytically based on the estimated density and LIM power spectra"

    def __init__(self, LIM_coefficients, LIM_correlation, Power_Spectrum_LIM, z, Lbox=600, Nbox=200, KIND=None, seed=1605, only_density =False):
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

            Pd = np.outer(Power_Spectrum_LIM._lingrowthd**2, LIM_correlation._PklinCF) [_iz,:]

            Pdinterp = interp1d(klist,Pd,fill_value=0.0,bounds_error=False)

            if only_density:
                pb = pbox.PowerBox(
                    N=self.Nbox,                     
                    dim=3,                     
                    pk = lambda k: Pdinterp(k), 
                    boxlength = self.Lbox,           
                    seed = self.seed,
                )

                self.deltamap = pb.delta_x() #density map, basis of this KIND of approach

            else:
                pb = pbox.PowerBox(
                    N=self.Nbox,                     
                    dim=3,                     
                    pk = lambda k: Pdinterp(k), 
                    boxlength = self.Lbox,           
                    seed = self.seed,
                )

                self.deltamap = pb.delta_x() #density map, basis of this KIND of approach

                #then we make a map of the linear LIM fluctuation
                PdLIM = Power_Spectrum_LIM._Pk_deltaLIM[_iz]
                
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
                    seed = self.seed+1,  # uncorrelated
                )

                self.LIMmapNL = self.Inu_global*pbe.delta_x()

                #and finally, just add them together!
                self.LIMmap_lin = self.LIMmaplin 
                self.LIMmap = self.LIMmaplin +  self.LIMmapNL


class CoevalBoxes_percell:

    "We produce a density map, transform each cell to SFRD "

    def __init__(self, which_SFRD, LIM_coefficients, LIM_correlations, Power_Spectrum_LIM, Zeus_coefficients, z, LineParams, AstroParams, HMFintclass, CosmoParams, Lbox=600, Nbox=200, seed=1605):

        start = time.time()
        zlist = LIM_coefficients.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.Inu_global = LIM_coefficients.Inu_avg[_iz]
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed
        self.z = zlist[_iz] #will be slightly different from z input

        analytical_box = CoevalBox_LIM_zeuslike(LIM_coefficients=LIM_coefficients, LIM_correlation=LIM_correlations, Power_Spectrum_LIM=Power_Spectrum_LIM, z=z, Lbox=Lbox, Nbox=Nbox, KIND=1, seed=seed, only_density =True)
        #print('done analytical in ' + str(time.time() - start))
        
        self.delta_box = analytical_box.deltamap

        klistfftx = np.fft.fftfreq(self.delta_box.shape[0],Lbox/Nbox)*2*np.pi
        klist3Dfft = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
        density_fft = np.fft.fftn(self.delta_box)

        smooth_density_fields = np.array(z21_utilities.tophat_smooth(LineParams._R, 
        klist3Dfft, density_fft) )

        # SarahLibanore: ionizing photons per cell 
        L_cell_Mpc = Lbox/Nbox #* u.Mpc
        sphere_FACTOR = 0.620350491 # factor relating cube length to filter radius = (4PI/3)^(-1/3)

        smooth_density_fields_cell = (np.array(z21_utilities.tophat_smooth(L_cell_Mpc * sphere_FACTOR, klist3Dfft, density_fft))).flatten()

        delta_crit_ST = CosmoParams.delta_crit_ST
        a_corr_EPS = CosmoParams.a_corr_EPS

        variance = np.var(smooth_density_fields_cell)
        sigmaR = ne.evaluate('sqrt(variance)')
        deltaArray = ne.evaluate('smooth_density_fields_cell * sigmaR')

        mArray, deltaArray_Mh = np.meshgrid(HMFintclass.Mhtab, deltaArray, indexing = 'ij', sparse = True)

        sigmaM = HMFintclass.sigmaintlog((np.log(mArray),z))

        modSigmaSq = ne.evaluate('sigmaM*sigmaM - sigmaR*sigmaR')
        indexTooBig = (modSigmaSq <= 0.0)
        modSigmaSq[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
        modSigma = ne.evaluate('sqrt(modSigmaSq)')

        nu0 = ne.evaluate('delta_crit_ST / sigmaM')
        nu0[indexTooBig] = 1.0
        modd = ne.evaluate('delta_crit_ST - deltaArray_Mh')
        nu = ne.evaluate('modd / modSigma')

        EPS_HMF_corr = ne.evaluate('(nu/nu0) * (sigmaM/modSigma)* (sigmaM/modSigma) * exp(-a_corr_EPS * (nu*nu-nu0*nu0)/2. ) * (1.0 + deltaArray_Mh)')
        #print('Done EPS corr in ' + str(time.time() - start))
        if which_SFRD == 'full':
            HMF_curr = np.exp(HMFintclass.logHMFint((np.log(mArray),z)))
            SFRtab_currII = sfrd.SFR_II(AstroParams, CosmoParams, HMFintclass, mArray, z, 0.)
            #print('SFR ok in ' + str(time.time() - start))
            integrand = EPS_HMF_corr *  HMF_curr * SFRtab_currII * HMFintclass.Mhtab[:,np.newaxis]

            if LineParams.LINE == 'Ha' and LineParams.LINE_MODEL == 'Nionrate':
                #Z = 0.2
                #Nion_rate = 10**(-0.0029*(np.log10(Z) + 7.3)**2.5 + 53.81 - np.log10(2.55))*SFRtab_currII # s-1 
                BMF_val = BMF(Zeus_coefficients, HMFintclass,CosmoParams,AstroParams)
                ir_use = idx = np.abs(CosmoParams._Rtabsmoo - Lbox/Nbox).argmin()

                Nion_rate = BMF_val.niondot_delta_r(smooth_density_fields_cell.flatten(),CosmoParams._Rtabsmoo[ir_use])
                Nion_rate = Nion_rate.reshape((Nbox,Nbox,Nbox,len(Zeus_coefficients.zintegral)))[:,:,:,_iz]
                #print(Nion_rate)

                frec = 0.45
                hP = 1.73094832e-60 # units of Lsun * s / Hz
                L_Halpha = hP * LineParams.nu_rest * frec * Nion_rate # Hz * s-1
                integrand_LIM = EPS_HMF_corr *  HMF_curr * L_Halpha * HMFintclass.Mhtab[:,np.newaxis]

            else:
                integrand_LIM = EPS_HMF_corr *  HMF_curr * LineLuminosity(SFRtab_currII, LineParams, AstroParams, CosmoParams, HMFintclass, mArray, z)  * HMFintclass.Mhtab[:,np.newaxis]
                #print('Integrand ok in ' + str(time.time() - start))

            SFRDbox_flattend = np.trapezoid(integrand, HMFintclass.logtabMh, axis = 0) 
            #print('SFRD ok in ' + str(time.time() - start))
            rhoLbox_flattened = np.trapezoid(integrand_LIM,HMFintclass.logtabMh, axis = 0)
            #print('rho box ok in ' + str(time.time() - start))
        elif which_SFRD == 'approx':
            if AstroParams.second_order_SFRD:
                SFRDbox_flattend = Zeus_coefficients.np.exp(Zeus_coefficients.gamma_II_index2D*deltaArray + Zeus_coefficients.gamma2_II_index2D*deltaArray**2)
            else:
                SFRDbox_flattend = np.exp(Zeus_coefficients.gamma_II_index2D*deltaArray) 

            SFRDbox_flattend *= Zeus_coefficients.SFRDbar2D_II

            # ?????????
            #rhoLbox_flattened = SFRDbox_flattend * ??? LineLuminosity

    #        integrand_LIM = EPS_HMF_corr *  HMF_curr * LineLuminosity(SFRtab_currII, LineParams, AstroParams, CosmoParams, HMFintclass, mArray, z)  * HMFintclass.Mhtab[:,np.newaxis]


        self.SFRD_box = SFRDbox_flattend.reshape(Nbox,Nbox,Nbox)

        rhoL_box = rhoLbox_flattened.reshape(Nbox,Nbox,Nbox)

        self.Inu_box = rhoL_box * LIM_coefficients.coeff1_LIM[_iz] 


def build_lightcone(which_lightcone,
             input_zvals,
             Lbox, 
             Ncell, 
             seed, 
             RSDMODE, 
             analytical, 
             LineParams, 
             AstroParams, 
             ClassyCosmo, 
             HMFintclass, 
             CosmoParams,      
             ZMIN,    
             include_label = ''    
             ):

    save_path = os.path.join(os.getcwd(), "oLIMpus")
    folder_out = os.path.abspath(os.path.join(save_path, "..", 'analysis_' + str(Lbox) + ',' + str(Ncell) ))

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    folder = folder_out + '/lightcones'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename_all = folder + which_lightcone + include_label + '.pkl'
    if os.path.exists(filename_all):
        with open(filename_all, 'rb') as handle:
            lightcone = pickle.load(handle)
            return lightcone

    zvals = input_zvals[::-1]

    z_long = np.linspace(zvals[0],zvals[-1],1000)

    correlations_21 = Correlations(CosmoParams, ClassyCosmo)   
    coefficients_21 = sfrd.get_T21_coefficients(CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
   
    PS21 = Power_Spectra(CosmoParams, AstroParams, ClassyCosmo, correlations_21, coefficients_21, RSD_MODE = RSDMODE)

    lightcone = np.zeros((Ncell, Ncell, len(z_long)))
    box = []
    for zi in zvals:
        box.append(lightcone_single_z(zi, zvals, Lbox,Ncell,seed,which_lightcone, analytical, coefficients_21, correlations_21, PS21, HMFintclass, ClassyCosmo,CosmoParams,AstroParams,LineParams,ZMIN,RSDMODE))
        
    lightcone[:, :, 0] = box[0][:, :, 0]        
    # Loop over each z in z_long
    for z_idx, zi in (enumerate(z_long[1:], start=1)):
        # Find which two matrices to interpolate between
        idx = np.searchsorted(zvals, zi) - 1
        idx = np.clip(idx, 0, len(zvals) - 2)  # Keep index within bounds
        
        z1, z2 = zvals[idx], zvals[idx + 1]
        mat1, mat2 = box[idx], box[idx + 1]
        
        # Interpolation weight
        w = (zi - z1) / (z2 - z1)
        
        # Interpolate between contiguous slices
        lightcone[:, :, z_idx] = (1 - w) * mat1[:, :, z_idx % Ncell] + w * mat2[:, :, z_idx % Ncell]

    lightcone[np.isnan(lightcone)] = 0.

    with open(filename_all, 'wb') as handle:
        pickle.dump(lightcone,handle)
    

    return lightcone


def lightcone_single_z(zi, zvals, Lbox,Ncell,seed,which_lightcone, analytical, coefficients_21, correlations_21, PS21, HMFintclass, ClassyCosmo,CosmoParams,AstroParams,LineParams,ZMIN,RSDMODE):

    if which_lightcone == 'T21':
        if analytical and zi == zvals[0]:
            print('Warning! The bubble part is not analytical')
        else:
            if zi == zvals[0]:
                print('Warning! The T21 map is only  analytical, except for the bubble part')

        BMF_val = BMF(coefficients_21, HMFintclass,CosmoParams,AstroParams)

        box = zeus21_maps.T21_bubbles(coefficients_21,PS21,zi,Lbox,Ncell,seed,correlations_21,CosmoParams,AstroParams,ClassyCosmo,BMF_val).T21map

    elif which_lightcone == 'density':
        if not analytical and zi == zvals[0]:
            print('Warning! The density map is only  analytical')

        BMF_val = BMF(coefficients_21, HMFintclass,CosmoParams,AstroParams)

        box = zeus21_maps.T21_bubbles(coefficients_21,PS21,zi,Lbox,Ncell,seed,correlations_21,CosmoParams,AstroParams,ClassyCosmo,BMF_val).deltamap

    elif which_lightcone == 'xHI':
        if analytical and zi == zvals[0]:
            print('Warning! The xHI map cannot be computed analytically')

        BMF_val = BMF(coefficients_21, HMFintclass,CosmoParams,AstroParams)

        box.append(zeus21_maps.T21_bubbles(coefficients_21,PS21,zi,Lbox,Ncell,seed,correlations_21,CosmoParams,AstroParams,ClassyCosmo,BMF_val).xHI_map)

    else:
        correlations = Correlations_LIM(LineParams, CosmoParams, ClassyCosmo)
        coefficients = get_LIM_coefficients(CosmoParams,  AstroParams, HMFintclass, LineParams, zmin=ZMIN)

        PSLIM = Power_Spectra_LIM(CosmoParams, AstroParams, LineParams, correlations, coefficients, RSD_MODE = RSDMODE)

        if which_lightcone == 'LIM':

            if analytical:
                box.append(CoevalBox_LIM_zeuslike(LIM_coefficients=coefficients, LIM_correlation=correlations, Power_Spectrum_LIM=PSLIM, z=zi, Lbox=Lbox, Nbox=Ncell, KIND=1, seed=seed).LIMmap)

            else:
                box.append(CoevalBoxes_percell('full',coefficients,correlations,PSLIM,coefficients_21,zi,LineParams,AstroParams,HMFintclass,CosmoParams,Lbox,Ncell,seed).Inu_box)

        elif which_lightcone == 'SFRD':
                if analytical and zi == zvals[0]:
                    print('Warning! The SFRD map cannot be computed analytically')

                box.append(CoevalBoxes_percell('full',coefficients,correlations,PSLIM,coefficients_21,zi,LineParams,AstroParams,HMFintclass,CosmoParams,Lbox,Ncell,seed).SFRD_box)

        else:
            print('Check lightcone')
            return 

    return box