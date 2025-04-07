"""

Make maps! For fun and science

Author: Julian B. Muñoz
UT Austin - August 2024

"""

from . import cosmology
from . import constants
from . import z21_utilities

import numpy as np
import powerbox as pbox
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from pyfftw import empty_aligned as empty
from tqdm import trange
import time

import astropy
from astropy import units as u


class CoevalMaps:
    "Class that calculates and keeps coeval maps, one z at a time."

    def __init__(self, T21_coefficients, Power_Spectrum, z, Lbox=600, Nbox=200, KIND=None, seed=1605):
        'the KIND flag determines the kind of map you make. Options are:'
        'KIND = 0, only T21 lognormal. OK approximation'
        'KIND = 1, density and T21 correlated. T21 has a gaussian and a lognormal component. Decent approximation'
        'KIND = 2, all maps'
        'KIND = 3, same as 2 but integrating over all R. Slow but most accurate'

        zlist = T21_coefficients.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.T21global = T21_coefficients.T21avg[_iz]
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed
        self.z = zlist[_iz] #will be slightly different from z input

        klist = Power_Spectrum.klist_PS
        k3over2pi2 = klist**3/(2*np.pi**2)


        if (KIND == 0): #just T21, ~gaussian
                
            P21 = Power_Spectrum.Deltasq_T21_lin[_iz]/k3over2pi2
            P21norminterp = interp1d(klist,P21/self.T21global**2,fill_value=0.0,bounds_error=False)


            pb = pbox.PowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: P21norminterp(k), 
                boxlength = self.Lbox,           
                seed = self.seed                
            )

            self.T21map = self.T21global * (1 + pb.delta_x() )
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

            #then we make a map of the linear T21 fluctuation, better to use the cross to keep sign, at linear level same 
            PdT21 = Power_Spectrum.Deltasq_dT21[_iz]/k3over2pi2

            powerratioint = interp1d(klist,PdT21/Pd,fill_value=0.0,bounds_error=False)

            deltak = pb.delta_k()

            powerratio = powerratioint(pb.k())
            T21lin_k = powerratio * deltak
            self.T21maplin= self.T21global + powerboxCtoR(pb,mapkin = T21lin_k)

            #now make a nonlinear correction, built as \sum_R [e^(gR dR) - gR dR]. Uncorrelatd with all dR so just a separate field!
            #NOTE: its not guaranteed to work, excess power can be negative in some cases! Not for each component xa, Tk, but yes for T21
            excesspower21 = (Power_Spectrum.Deltasq_T21[_iz,:]-Power_Spectrum.Deltasq_T21_lin[_iz,:])/k3over2pi2

            lognormpower = interp1d(klist,excesspower21/self.T21global**2,fill_value=0.0,bounds_error=False)
            #G or logG? TODO revisit
            pbe = pbox.LogNormalPowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: lognormpower(k), 
                boxlength = self.Lbox,           
                seed = self.seed+1                # uncorrelated
            )

            self.T21mapNL = self.T21global*pbe.delta_x()

            #and finally, just add them together!
            self.T21map = self.T21maplin +  self.T21mapNL



        else:
            print('ERROR, KIND not implemented yet!')


def make_ion_fields(CosmoParams, AstroParams, CoeffStructure, ClassyCosmo, CorrFClass, BMF, input_z, boxlength=300., ncells=300, seed=1234, r_precision=1., timer=True, logd = False, barrier = None, spherize=False, FLAG_return_densities = 0):
    """
    Generates a 3D map of ionized fields and ionized fraction of hydrogen.
    
    Uses a density threshold barrier determined from a converged bubble mass function. With default parameters, the code takes about 20 minutes on laptop to run.

    Parameters
    ----------
    CosmoParams: zeus21.Cosmo_Parameters class
        Stores cosmology.
    CoeffStructure: zeus21.get_T21_coefficients class
        Stores sfrd and 21cm coefficients.
    ClassyCosmo: zeus21.runclass class
        Sets up Class cosmology.
    CorrFClass: zeus21.Correlations class
        Calculates correlation functions.
    BMF: zeus21.reionization class
        Computes bubble mass functions and barriers.
    input_z: 1D np.array
        The redshifts at which to compute output maps. Narrowed down later to select available redshifts from CoeffStructure.zintegral.
    boxlength: float
        Comoving physical side length of the box. Default is 300 cMpc.
    ncells: int
        Number of cells on a side. Default is 300 cells.
    seed: int
        Sets the predetermined generation of maps. Default is 1234.
    r_precision: float
        Allows to change the steps of the radii for faster computation. Default (and max) is 1, lower values make the computation faster at the cost of accuracy.
    timer: bool
        Whether to print the time elapsed along the process. Default is True.
    logd: bool
        Whether to use lognormal (True) or Gaussian (False) density fields. Default is False.
    barrier: function
        Input density barrier to be used as the threshold for map generation. Takes z index //\\Change//\\ (relative to CoeffStructure.zintegral) as input and returns np.array of shape of radii. Default is None.
    spherize: bool
        Whether to flag spheres around ionized cells (True) instead of only central pixel flagging (False). Default is False. Central pixel flagging is generally more consistent with the bubble mass function than spherizing.
    FLAG_return_densities: int
        Options: (0, 1, 2). Default is 0.
            0: returns only the ionized fields and ionized fractions.
            1: returns only the ionized fields, ionized fractions, and density field at the last redshift.
            2: returns the ionized fields, ionized fractions, density field at the last redshift, and the smoothed density fields at each scale at the last redshift.

    Outputs
    ----------
    ion_fields: 4D np.array
        Resultant 3D ionized field maps at each redshift. The first dimension is redshifts, and the three other dimensions are spatial.
    ion_frac: 1D np.array
        Ionized fraction at each redshift in ion_fields.
    Optional:
        density_field: 3D np.array
            The field of densities at the last redshift.
        smoothed_density_fields: 4D np.array
            The field of densities at the last redshift, smoothed over each radius scale. The first dimension is the smoothing scales, the other three are spatial.
    
    """
    
    #Measure time elapsed from start
    start_time = time.time()
    
    if timer:
        z21_utilities.print_timer(start_time, 'Making density field...')

    #selecting redshifts and radii from available redshifts
    zlist = CoeffStructure.zintegral
    z_idx = z21_utilities.find_nearest_idx(zlist, input_z)
    z = zlist[z_idx]

    Rs = CoeffStructure.Rtabsmoo
    r_idx = np.linspace(0,len(Rs)-1,int(len(Rs)*r_precision),dtype=int)
    r = Rs[r_idx]
    dx = boxlength/ncells

    #Generating matter power spectrum at z=5
    klist = CorrFClass._klistCF
    pk_matter = np.zeros_like(klist)
    for i, k in enumerate(klist):
        pk_matter[i] = ClassyCosmo.pk(k, z[0])
    pk_spl = spline(np.log(klist), np.log(pk_matter))

    #generating density map
    if logd:
        pb = pbox.LogNormalPowerBox(N=ncells, dim=3, pk=(lambda k: np.exp(pk_spl(np.log(k)))), boxlength=boxlength, seed=seed)
    else:
        pb = pbox.PowerBox(N=ncells, dim=3, pk=(lambda k: np.exp(pk_spl(np.log(k)))), boxlength=boxlength, seed=seed)
    density_field = pb.delta_x()

    if timer:
        z21_utilities.print_timer(start_time, 'Creating smoothing function...')

    #comment
    klistfftx = np.fft.fftfreq(density_field.shape[0],dx)*2*np.pi
    klist3Dfft = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
    density_fft = np.fft.fftn(density_field)

    if timer:
        z21_utilities.print_timer(start_time, 'Smoothing...')

    #comment
    smooth_density_fields = np.array([z21_utilities.tophat_smooth(rr, 
    klist3Dfft, density_fft) for rr in r])

    # SarahLibanore: ionizing photons per cell 
    L_cell_Mpc = boxlength/ncells * u.Mpc
    sphere_FACTOR = 0.620350491 # factor relating cube length to filter radius = (4PI/3)^(-1/3)

    smooth_density_fields_cell = (np.array(z21_utilities.tophat_smooth(L_cell_Mpc.value * sphere_FACTOR, 
    klist3Dfft, density_fft))).flatten()

    idx_cell = np.abs(r - L_cell_Mpc.value * sphere_FACTOR).argmin()
    z_idx_cell = np.abs(zlist - z).argmin()

    n_ion_photon_map = BMF.nion_delta_r_int(smooth_density_fields_cell, idx_cell)
    n_rec_map = BMF.nrec(smooth_density_fields_cell, BMF.ion_frac)

    if timer:
        z21_utilities.print_timer(start_time, 'Creating ionized field...')

    if barrier is None:
        barrier = BMF.barrier

    ion_fields = []
    xHI_map = []
    ion_frac = np.zeros(len(z))
    for i in range(len(z)):
        curr_z_idx = z_idx[i]
        ion_field = ionize(CosmoParams, zlist, Rs, curr_z_idx, smooth_density_fields, barrier, r_idx, klist3Dfft, spherize)
        ion_fields.append(ion_field)
        ion_frac[i] = np.sum(ion_field)/ncells**3
        
        # SarahLibanore: xHI map 
        xHI_map = np.maximum(0,np.minimum(1,(1.+n_rec_map[:,z_idx_cell] - n_ion_photon_map[:,z_idx_cell])))
    
    ion_fields = np.array(ion_fields)
    xHI_map = np.array(xHI_map).reshape((ncells,ncells,ncells))


    if timer:
        print('Done!')
        print('\nTotal time:')
        z21_utilities.print_timer(start_time)

    if FLAG_return_densities == 0:
        return ion_fields, ion_frac, xHI_map

    elif FLAG_return_densities == 1:
        return ion_fields, ion_frac, density_field, xHI_map

    elif FLAG_return_densities == 2:
        return ion_fields, ion_frac, density_field, smooth_density_fields, xHI_map

    else:
        print('WARNING: FLAG_return_densities is not set to (0, 1, or 2). Defaulting to 0.')
        return ion_fields, ion_frac, xHI_map


#look over this again
def ionize(CosmoParams, zlist, Rs, curr_z_idx, smooth_density_fields, barrier, r_idx, klist3Dfft, spherize):

    Dg0 = CosmoParams.growthint(zlist[0])
    Dg = CosmoParams.growthint(zlist[curr_z_idx])
    if not spherize:
        ion_field = np.any(smooth_density_fields > (Dg0/Dg)*barrier[curr_z_idx, r_idx][:, None, None, None], axis=0)
    else:
        ion_field_Rs = []
        for j in range(len(r_idx)):
            curr_R = r_idx[j]
            ion_field_oneR = smooth_density_fields[j] > (Dg0/Dg)*barrier[curr_z_idx, curr_R]
            ionk = np.fft.fftn(ion_field_oneR)
            cutoff = 1/(4/3*np.pi*Rs[curr_R]**3)/2*(1+barrier[curr_z_idx, curr_R]) #comment
            ion_spheres = z21_utilities.tophat_smooth(Rs[curr_R]/(1+barrier[curr_z_idx, curr_R])**(1/3), klist3Dfft, ionk) > cutoff
            ion_field_Rs.append(ion_spheres)
                
        ion_field = np.any(ion_field_Rs, axis=0)
    return ion_field



def powerboxCtoR(pbobject,mapkin = None):
    'Function to convert a complex field to real 3D (eg density, T21...) on the powerbox notation'
    'Takes a powerbox object pbobject, and a map in k space (mapkin), or otherwise assumes its pbobject.delta_k() (tho in that case it should be delta_x() so...'

    realmap = empty((pbobject.N,) * pbobject.dim, dtype='complex128')
    if (mapkin is None):
        realmap[...] = pbobject.delta_k()
    else:
        realmap[...] = mapkin
    realmap[...] = pbobject.V * pbox.dft.ifft(realmap, L=pbobject.boxlength, a=pbobject.fourier_a, b=pbobject.fourier_b)[0]
    realmap = np.real(realmap)

    return realmap


# SarahLibanore: Temperature map using the local xHI instead of the average one
class T21_bubbles:

    def __init__(self, T21_coefficients, Power_Spectrum, z, Lbox, Nbox, seed, CorrFClass, CosmoParams, AstroParams, ClassyCosmo, BMF):

        zlist = T21_coefficients.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.T21global = T21_coefficients.T21avg[_iz]
        self.xHavg = T21_coefficients.xHI_avg[_iz]
        self.T21global_scaled = self.T21global / self.xHavg

        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed
        self.z = zlist[_iz] #will be slightly different from z input

        klist = Power_Spectrum.klist_PS
        k3over2pi2 = klist**3/(2*np.pi**2)

        temp = make_ion_fields(CosmoParams, AstroParams, T21_coefficients, ClassyCosmo, CorrFClass, BMF, z, boxlength=Lbox, ncells=Nbox, seed=seed, r_precision=1., timer=False, logd = False, barrier = None, spherize=False, FLAG_return_densities = 0)
        self.xHI_map = temp[-1] * (1-temp[0][0])

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

        #then we make a map of the linear T21 fluctuation, better to use the cross to keep sign, at linear level same 
        PdT21_scaled = Power_Spectrum.Deltasq_dT21[_iz]/k3over2pi2 / self.xHavg

        powerratioint = interp1d(klist,PdT21_scaled/Pd,fill_value=0.0,bounds_error=False)

        deltak = pb.delta_k()

        powerratio = powerratioint(pb.k())
        T21lin_k = powerratio * deltak 
        self.T21maplin= self.T21global_scaled + powerboxCtoR(pb,mapkin = T21lin_k) 

        #now make a nonlinear correction, built as \sum_R [e^(gR dR) - gR dR]. Uncorrelatd with all dR so just a separate field!
        #NOTE: its not guaranteed to work, excess power can be negative in some cases! Not for each component xa, Tk, but yes for T21
        excesspower21_scaled = (Power_Spectrum.Deltasq_T21[_iz,:] - Power_Spectrum.Deltasq_T21_lin[_iz,:])/k3over2pi2 / self.xHavg**2

        lognormpower = interp1d(klist,excesspower21_scaled/self.T21global_scaled**2,fill_value=0.0,bounds_error=False)
        #G or logG? TODO revisit
        pbe = pbox.LogNormalPowerBox(
            N=self.Nbox,                     
            dim=3,                     
            pk = lambda k: lognormpower(k), 
            boxlength = self.Lbox,           
            seed = self.seed+1                # uncorrelated
        )

        self.T21mapNL = self.T21global_scaled*pbe.delta_x()

        #and finally, just add them together!
        self.T21map = self.T21maplin +  self.T21mapNL

        self.T21map *= self.xHI_map



